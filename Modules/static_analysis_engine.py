from __future__ import annotations

import binascii
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)


class RuleMatcher(Protocol):
    def match(self, *, data: bytes) -> Sequence[Any]:
        ...


class ProcessScanner(Protocol):
    """Optional adapter interface for host applications.

    The engine is intentionally UI-agnostic. If the host app can supply a
    scanner object with some or all of these methods, the engine will use them.
    Missing methods are handled gracefully.
    """

    def detect_process_hollowing(self, process_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ...

    def detect_unusual_relationships(self) -> List[Dict[str, Any]]:
        ...

    def detect_persistence_methods(self) -> List[Dict[str, Any]]:
        ...

    def get_extended_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        ...


@dataclass
class StaticFinding:
    id: str
    category: str
    finding_type: str
    severity: str
    target: str
    location: str
    description: str
    process_id: Optional[int] = None
    process_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "static_analysis_engine"


@dataclass
class StaticAnalysisSummary:
    processes_scanned: int = 0
    processes_skipped: int = 0
    memory_regions_scanned: int = 0
    threats_found: int = 0
    scan_duration_seconds: float = 0.0
    findings_by_severity: Dict[str, int] = field(default_factory=dict)
    findings_by_category: Dict[str, int] = field(default_factory=dict)


@dataclass
class RuleStatus:
    loaded: bool
    compiled: bool
    rule_count: int = 0
    error: Optional[str] = None


class StaticAnalysisEngine:
    PAGE_EXECUTE = 0x10
    PAGE_EXECUTE_READ = 0x20
    PAGE_EXECUTE_READWRITE = 0x40
    PAGE_EXECUTE_WRITECOPY = 0x80
    PAGE_NOACCESS = 0x01
    PAGE_READONLY = 0x02
    PAGE_READWRITE = 0x04
    PAGE_WRITECOPY = 0x08
    PAGE_TARGETS_INVALID = 0x40000000
    PAGE_TARGETS_NO_UPDATE = 0x40000000

    DEFAULT_INJECTION_PATTERNS: Dict[str, bytes] = {
        "shellcode": rb"\x55\x8B\xEC|\x90{4,}",
        "script_injection": rb"(eval|exec|system|subprocess\.run)",
        "memory_manipulation": rb"(VirtualAlloc|WriteProcessMemory)",
        "dll_injection": rb"(LoadLibrary|GetProcAddress)",
        "code_execution": rb"(WScript\.Shell|cmd\.exe|powershell\.exe)",
        "encoded_commands": rb"([A-Za-z0-9+/]{40,}={0,2})",
    }

    SHELLCODE_PATTERNS: Dict[str, bytes] = {
        "nop_sled": rb"\x90{5,}",
        "function_prolog": rb"\x55\x8B\xEC",
        "syscall_patterns": rb"\xCD\x80|\x0F\x34|\x0F\x05",
        "egg_hunter": rb"\x66\x81\xCA\xFF\x0F\x42\x52\x6A\x02",
        "jump_call_pop": rb"\xEB.\xE8",
        "api_hashing": rb"\x74\x0C\x75",
        "peb_access": rb"\x64\xA1\x30\x00\x00\x00|\x64\x8B\x1D\x30\x00\x00\x00",
    }

    SUSPICIOUS_PROCESS_NAMES = {
        "cmd.exe",
        "powershell.exe",
        "wscript.exe",
        "cscript.exe",
        "rundll32.exe",
    }

    CRITICAL_PROCESS_NAMES = {
        "explorer.exe",
        "svchost.exe",
        "lsass.exe",
        "winlogon.exe",
        "csrss.exe",
        "services.exe",
        "smss.exe",
        "wininit.exe",
        "system",
    }

    def __init__(
        self,
        *,
        yara_matcher: Optional[RuleMatcher] = None,
        scanner: Optional[ProcessScanner] = None,
        injection_patterns: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.yara_matcher = yara_matcher
        self.scanner = scanner
        self.injection_patterns = injection_patterns or dict(self.DEFAULT_INJECTION_PATTERNS)
        self.findings: List[StaticFinding] = []
        self.last_summary = StaticAnalysisSummary()

    # ---------------------------------------------------------------------
    # Rule status / initialization
    # ---------------------------------------------------------------------
    def get_rule_status(self) -> RuleStatus:
        loaded = self.yara_matcher is not None
        compiled = loaded
        rule_count = 1 if loaded else 0
        return RuleStatus(loaded=loaded, compiled=compiled, rule_count=rule_count)

    # ---------------------------------------------------------------------
    # Finding helpers
    # ---------------------------------------------------------------------
    def build_detection(
        self,
        *,
        category: str,
        finding_type: str,
        severity: str,
        target: str,
        location: str,
        description: str,
        process_id: Optional[int] = None,
        process_name: str = "",
        details: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> StaticFinding:
        finding = StaticFinding(
            id=str(uuid.uuid4()),
            category=category,
            finding_type=finding_type,
            severity=severity.upper(),
            target=target,
            location=location,
            description=description,
            process_id=process_id,
            process_name=process_name,
            details=details or {},
            evidence=evidence or [],
        )
        self.findings.append(finding)
        return finding

    def _severity_from_risk(self, risk_level: int) -> str:
        if risk_level >= 80:
            return "CRITICAL"
        if risk_level >= 60:
            return "HIGH"
        if risk_level >= 30:
            return "MEDIUM"
        return "LOW"

    def summarize_findings(self, start_time: float, processes_scanned: int, processes_skipped: int, memory_regions_scanned: int) -> StaticAnalysisSummary:
        by_severity: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for finding in self.findings:
            by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
            by_category[finding.category] = by_category.get(finding.category, 0) + 1

        self.last_summary = StaticAnalysisSummary(
            processes_scanned=processes_scanned,
            processes_skipped=processes_skipped,
            memory_regions_scanned=memory_regions_scanned,
            threats_found=len(self.findings),
            scan_duration_seconds=time.time() - start_time,
            findings_by_severity=by_severity,
            findings_by_category=by_category,
        )
        return self.last_summary

    # ---------------------------------------------------------------------
    # Memory protection analysis
    # ---------------------------------------------------------------------
    def classify_memory_region(self, protection: int) -> List[str]:
        labels: List[str] = []
        if protection & self.PAGE_EXECUTE:
            labels.append("EXECUTE")
        if protection & self.PAGE_EXECUTE_READ:
            labels.append("EXECUTE_READ")
        if protection & self.PAGE_EXECUTE_READWRITE:
            labels.append("EXECUTE_READWRITE")
        if protection & self.PAGE_EXECUTE_WRITECOPY:
            labels.append("EXECUTE_WRITECOPY")
        if protection & self.PAGE_NOACCESS:
            labels.append("NOACCESS")
        if protection & self.PAGE_READONLY:
            labels.append("READONLY")
        if protection & self.PAGE_READWRITE:
            labels.append("READWRITE")
        if protection & self.PAGE_WRITECOPY:
            labels.append("WRITECOPY")
        if protection & self.PAGE_TARGETS_INVALID:
            labels.append("CFG_INVALID")
        return labels

    def analyze_memory_protection(
        self,
        *,
        pid: int,
        process_name: str,
        protection: int,
        base_addr: int,
        region_size: int,
    ) -> Optional[StaticFinding]:
        location = f"Memory region at {hex(base_addr)}, size: {region_size}"

        if protection & self.PAGE_TARGETS_NO_UPDATE and process_name.lower() in self.CRITICAL_PROCESS_NAMES:
            return self.build_detection(
                category="cfg_bypass",
                finding_type="Control Flow Guard Bypass",
                severity="HIGH",
                target=process_name,
                location=location,
                description=f"PAGE_TARGETS_NO_UPDATE detected at {hex(base_addr)}",
                process_id=pid,
                process_name=process_name,
                details={"protection": protection},
            )

        if protection & self.PAGE_EXECUTE_READWRITE:
            return self.build_detection(
                category="suspicious_memory",
                finding_type="Suspicious Memory Protection",
                severity="HIGH",
                target=process_name,
                location=location,
                description="RWX Memory (Read-Write-Execute)",
                process_id=pid,
                process_name=process_name,
                details={"protection": protection},
            )

        if (protection & self.PAGE_EXECUTE) and (protection & self.PAGE_READWRITE) and not (protection & self.PAGE_READONLY):
            return self.build_detection(
                category="wx_memory",
                finding_type="High Risk Memory Protection",
                severity="HIGH",
                target=process_name,
                location=location,
                description="Write+Execute Memory (No Read Permission)",
                process_id=pid,
                process_name=process_name,
                details={"protection": protection},
            )

        return None

    def summarize_memory_protections(self, regions: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        summary = {
            "executable_regions": 0,
            "execute_read_regions": 0,
            "readonly_regions": 0,
            "noaccess_regions": 0,
            "writecopy_regions": 0,
            "rwx_regions": 0,
            "wx_regions": 0,
            "cfg_invalid_regions": 0,
        }
        for region in regions:
            protection = int(region.get("Protect", 0) or 0)
            if protection & self.PAGE_EXECUTE:
                summary["executable_regions"] += 1
            if protection & self.PAGE_EXECUTE_READ:
                summary["execute_read_regions"] += 1
            if protection & self.PAGE_READONLY:
                summary["readonly_regions"] += 1
            if protection & self.PAGE_NOACCESS:
                summary["noaccess_regions"] += 1
            if protection & self.PAGE_WRITECOPY:
                summary["writecopy_regions"] += 1
            if protection & self.PAGE_EXECUTE_READWRITE:
                summary["rwx_regions"] += 1
            if (protection & self.PAGE_EXECUTE) and (protection & self.PAGE_READWRITE) and not (protection & self.PAGE_READONLY):
                summary["wx_regions"] += 1
            if protection & self.PAGE_TARGETS_INVALID:
                summary["cfg_invalid_regions"] += 1
        return summary

    # ---------------------------------------------------------------------
    # Content scanning
    # ---------------------------------------------------------------------
    def scan_with_injection_patterns(
        self,
        memory_content: bytes,
        *,
        pid: int,
        process_name: str,
        base_addr: int,
        region_size: int,
    ) -> List[StaticFinding]:
        results: List[StaticFinding] = []
        location = f"Memory region at {hex(base_addr)}, size: {region_size}"

        for pattern_name, pattern in self.injection_patterns.items():
            try:
                matched = False
                if isinstance(pattern, bytes):
                    matched = re.search(pattern, memory_content, re.DOTALL) is not None
                elif hasattr(pattern, "search"):
                    matched = bool(pattern.search(memory_content))
                elif isinstance(pattern, str):
                    matched = re.search(pattern.encode(), memory_content, re.DOTALL) is not None

                if not matched:
                    continue

                results.append(
                    self.build_detection(
                        category="injection_patterns",
                        finding_type=f"Injection Pattern: {pattern_name}",
                        severity="MEDIUM",
                        target=process_name,
                        location=location,
                        description=f"Found at {hex(base_addr)}",
                        process_id=pid,
                        process_name=process_name,
                        details={"pattern_name": pattern_name, "region_size": region_size},
                    )
                )
            except Exception as exc:
                logger.debug("Error scanning for %s: %s", pattern_name, exc)
        return results

    def scan_with_yara(
        self,
        memory_content: bytes,
        *,
        pid: int,
        process_name: str,
        base_addr: int,
        region_size: int,
        extended_info: Optional[Dict[str, Any]] = None,
    ) -> List[StaticFinding]:
        if not self.yara_matcher:
            return []

        findings: List[StaticFinding] = []
        try:
            matches = self.yara_matcher.match(data=memory_content)
        except Exception as exc:
            logger.debug("YARA scanning error: %s", exc)
            return []

        for match in matches:
            rule_name = getattr(match, "rule", str(match))
            findings.append(
                self.build_detection(
                    category="yara_matches",
                    finding_type=f"YARA Rule: {rule_name}",
                    severity="HIGH",
                    target=process_name,
                    location=f"Memory region at {hex(base_addr)}, size: {region_size}",
                    description=f"YARA match at {hex(base_addr)}",
                    process_id=pid,
                    process_name=process_name,
                    details={"rule": rule_name, "extended_info": extended_info or {}},
                )
            )
        return findings

    def detect_shellcode_fragments(
        self,
        memory_content: bytes,
        *,
        base_address: int = 0,
        region_name: str = "memory_region",
    ) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []

        for pattern_name, pattern in self.SHELLCODE_PATTERNS.items():
            try:
                for match in re.finditer(pattern, memory_content, re.DOTALL):
                    start = match.start()
                    end = min(len(memory_content), match.end() + 32)
                    context = memory_content[max(0, start - 16):end]
                    fragment = memory_content[start:match.end()]
                    findings.append(
                        {
                            "pattern": pattern_name,
                            "offset": base_address + start,
                            "length": len(fragment),
                            "sha256": hashlib.sha256(fragment).hexdigest(),
                            "region_name": region_name,
                            "hex_preview": binascii.hexlify(context[:64]).decode("utf-8", errors="ignore"),
                        }
                    )
            except Exception as exc:
                logger.debug("Shellcode pattern scan failed for %s: %s", pattern_name, exc)
        return findings

    def identify_shellcode_techniques(self, code_bytes: bytes) -> List[str]:
        techniques: List[str] = []
        technique_patterns: List[Tuple[bytes, str]] = [
            (rb"\x90{5,}", "NOP sled"),
            (rb"\xeb.\xe8", "JMP/CALL/POP decoder"),
            (rb"\x31\xc9.*\xfe\xc1.*\x80", "XOR decoder loop"),
            (rb"\x64\xa1\x30\x00\x00\x00", "PEB access"),
            (rb"\x64\x8b\x1d\x30\x00\x00\x00", "PEB access (alt)"),
            (rb"\x6b\x65\x72\x6e\x65\x6c\x33\x32", "kernel32 string"),
            (rb"\x6e\x74\x64\x6c\x6c", "ntdll string"),
        ]
        for pattern, name in technique_patterns:
            try:
                if re.search(pattern, code_bytes, re.DOTALL):
                    techniques.append(name)
            except re.error:
                continue

        if self._has_stack_strings(code_bytes):
            techniques.append("Stack strings")
        if self._has_pic_indicators(code_bytes):
            techniques.append("Position-independent code")
        if self._has_api_hashing(code_bytes):
            techniques.append("API hashing")
        return techniques

    def _has_stack_strings(self, code_bytes: bytes) -> bool:
        push_sequence = 0
        for i in range(len(code_bytes) - 5):
            if code_bytes[i] == 0x68:
                dword = int.from_bytes(code_bytes[i + 1:i + 5], byteorder="little")
                if all(0x20 <= ((dword >> (8 * j)) & 0xFF) <= 0x7E for j in range(4)):
                    push_sequence += 1
                    if push_sequence >= 2:
                        return True
            else:
                push_sequence = 0
        return False

    def _has_pic_indicators(self, code_bytes: bytes) -> bool:
        for i in range(len(code_bytes) - 6):
            if code_bytes[i] == 0xE8 and code_bytes[i + 5] in (0x58, 0x59):
                return True
        getpc_patterns = [
            b"\xe8\x00\x00\x00\x00\x58",
            b"\xe8\x00\x00\x00\x00\x59",
            b"\xd9\xee\xd9\x74\x24\xf4",
            b"\xeb\x03\x5e\xeb\x05",
        ]
        return any(pattern in code_bytes for pattern in getpc_patterns)

    def _has_api_hashing(self, code_bytes: bytes) -> bool:
        hash_patterns = [
            rb"\x33\xc0[\x00-\xff]{0,6}\xac[\x00-\xff]{0,6}\xc1[\x00-\xff]{0,6}\x03",
            rb"\x74\x0c\x81\xec[\x00-\xff]{2}\x00\x00\xe8[\x00-\xff]{3}\x00\x00",
        ]
        return any(re.search(pattern, code_bytes, re.DOTALL) for pattern in hash_patterns)

    def analyze_shellcode_techniques(
        self,
        memory_content: bytes,
        *,
        pid: int,
        process_name: str,
        base_addr: int,
        region_size: int,
    ) -> List[StaticFinding]:
        findings: List[StaticFinding] = []
        techniques = self.identify_shellcode_techniques(memory_content)
        for technique in techniques:
            findings.append(
                self.build_detection(
                    category="shellcode_patterns",
                    finding_type=technique,
                    severity="HIGH",
                    target=process_name,
                    location=f"Memory region at {hex(base_addr)}, size: {region_size}",
                    description=f"Shellcode technique detected: {technique}",
                    process_id=pid,
                    process_name=process_name,
                    details={"base_addr": base_addr, "region_size": region_size},
                )
            )
        return findings

    def scan_memory_bytes(
        self,
        memory_content: bytes,
        *,
        pid: int,
        process_name: str,
        base_addr: int,
        region_size: int,
        extended_info: Optional[Dict[str, Any]] = None,
    ) -> List[StaticFinding]:
        findings: List[StaticFinding] = []
        findings.extend(
            self.scan_with_injection_patterns(
                memory_content,
                pid=pid,
                process_name=process_name,
                base_addr=base_addr,
                region_size=region_size,
            )
        )
        findings.extend(
            self.scan_with_yara(
                memory_content,
                pid=pid,
                process_name=process_name,
                base_addr=base_addr,
                region_size=region_size,
                extended_info=extended_info,
            )
        )

        fragments = self.detect_shellcode_fragments(memory_content, base_address=base_addr)
        if fragments:
            findings.append(
                self.build_detection(
                    category="shellcode_patterns",
                    finding_type="Shellcode Fragment(s)",
                    severity="HIGH",
                    target=process_name,
                    location=f"Memory region at {hex(base_addr)}, size: {region_size}",
                    description=f"Detected {len(fragments)} shellcode-like fragment(s)",
                    process_id=pid,
                    process_name=process_name,
                    evidence=fragments,
                )
            )
            findings.extend(
                self.analyze_shellcode_techniques(
                    memory_content,
                    pid=pid,
                    process_name=process_name,
                    base_addr=base_addr,
                    region_size=region_size,
                )
            )

        return findings

    # ---------------------------------------------------------------------
    # Process / registry / persistence hooks
    # ---------------------------------------------------------------------
    def analyze_process_hollowing(self, process_info: Dict[str, Any]) -> Optional[StaticFinding]:
        if not self.scanner or not hasattr(self.scanner, "detect_process_hollowing"):
            return None
        try:
            result = self.scanner.detect_process_hollowing(process_info)
        except Exception as exc:
            logger.debug("Process hollowing check failed: %s", exc)
            return None

        if not result or not result.get("executable_found", False):
            return None

        pid = int(process_info.get("pid", 0) or 0)
        process_name = str(process_info.get("name", "Unknown"))
        return self.build_detection(
            category="process_hollowing",
            finding_type="Process Hollowing",
            severity="HIGH",
            target=process_name,
            location="Process image / memory mismatch",
            description="Suspected process hollowing detected",
            process_id=pid,
            process_name=process_name,
            details=result,
        )

    def analyze_unusual_relationships(self) -> List[StaticFinding]:
        if not self.scanner or not hasattr(self.scanner, "detect_unusual_relationships"):
            return []
        findings: List[StaticFinding] = []
        try:
            relationships = self.scanner.detect_unusual_relationships()
        except Exception as exc:
            logger.debug("Error detecting unusual relationships: %s", exc)
            return []

        for relation in relationships:
            findings.append(
                self.build_detection(
                    category="process_relationships",
                    finding_type="Unusual Process Relationship",
                    severity="MEDIUM",
                    target=relation.get("child_name", "Unknown"),
                    location="Process hierarchy",
                    description=relation.get("description", "Unusual parent-child relationship detected"),
                    process_id=relation.get("child_pid", 0),
                    process_name=relation.get("child_name", "Unknown"),
                    details=relation,
                )
            )
        return findings

    def analyze_persistence_methods(self) -> List[StaticFinding]:
        if not self.scanner or not hasattr(self.scanner, "detect_persistence_methods"):
            return []
        findings: List[StaticFinding] = []
        try:
            persistence_findings = self.scanner.detect_persistence_methods()
        except Exception as exc:
            logger.debug("Error detecting persistence methods: %s", exc)
            return []

        for finding in persistence_findings:
            findings.append(
                self.build_detection(
                    category="persistence",
                    finding_type="Persistence Mechanism",
                    severity="HIGH",
                    target=finding.get("name", "Unknown"),
                    location=finding.get("location", "Registry or filesystem"),
                    description=finding.get("description", "Unknown persistence method"),
                    process_id=0,
                    process_name=finding.get("name", "Unknown"),
                    details=finding,
                )
            )
        return findings

    def normalize_registry_findings(self, registry_findings: Iterable[Dict[str, Any]]) -> List[StaticFinding]:
        findings: List[StaticFinding] = []
        for finding in registry_findings:
            findings.append(
                self.build_detection(
                    category="suspicious_registry",
                    finding_type="Registry Modification",
                    severity=str(finding.get("severity", "MEDIUM")),
                    target=finding.get("name", "registry"),
                    location=finding.get("location", "Registry"),
                    description=finding.get("description", "Suspicious registry change detected"),
                    details=dict(finding),
                )
            )
        return findings

    # ---------------------------------------------------------------------
    # High-level orchestration
    # ---------------------------------------------------------------------
    def run_static_analysis(
        self,
        process_regions: Iterable[Dict[str, Any]],
        *,
        registry_findings: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Tuple[List[StaticFinding], StaticAnalysisSummary]:
        self.findings = []
        start_time = time.time()
        processes_scanned = 0
        processes_skipped = 0
        memory_regions_scanned = 0

        for process in process_regions:
            pid = int(process.get("pid", 0) or 0)
            process_name = str(process.get("name", "Unknown"))
            if not pid:
                processes_skipped += 1
                continue

            processes_scanned += 1
            self.analyze_process_hollowing(process)

            for region in process.get("regions", []):
                memory_regions_scanned += 1
                protection = int(region.get("Protect", 0) or 0)
                base_addr = int(region.get("BaseAddress", 0) or 0)
                region_size = int(region.get("RegionSize", 0) or 0)
                memory_content = region.get("bytes", b"") or b""
                if not isinstance(memory_content, (bytes, bytearray)):
                    continue

                self.analyze_memory_protection(
                    pid=pid,
                    process_name=process_name,
                    protection=protection,
                    base_addr=base_addr,
                    region_size=region_size,
                )

                extended_info = None
                if self.scanner and hasattr(self.scanner, "get_extended_process_info"):
                    try:
                        extended_info = self.scanner.get_extended_process_info(pid)
                    except Exception as exc:
                        logger.debug("Failed to get extended info for %s: %s", process_name, exc)

                self.scan_memory_bytes(
                    memory_content,
                    pid=pid,
                    process_name=process_name,
                    base_addr=base_addr,
                    region_size=region_size,
                    extended_info=extended_info,
                )

        self.analyze_unusual_relationships()
        self.analyze_persistence_methods()
        if registry_findings:
            self.normalize_registry_findings(registry_findings)

        summary = self.summarize_findings(
            start_time,
            processes_scanned=processes_scanned,
            processes_skipped=processes_skipped,
            memory_regions_scanned=memory_regions_scanned,
        )
        return list(self.findings), summary

    # ---------------------------------------------------------------------
    # HadesAI integration helpers
    # ---------------------------------------------------------------------
    def to_threat_finding_payload(self, finding: StaticFinding) -> Dict[str, Any]:
        snippet = ""
        if finding.evidence:
            snippet = json.dumps(finding.evidence[:3], ensure_ascii=False)[:500]
        elif finding.details:
            snippet = json.dumps(finding.details, ensure_ascii=False)[:500]

        return {
            "path": finding.location,
            "threat_type": finding.finding_type,
            "pattern": finding.category,
            "severity": finding.severity,
            "code_snippet": snippet,
            "browser": finding.process_name or finding.target or "static-analysis",
            "context": finding.description,
        }


__all__ = [
    "RuleStatus",
    "StaticAnalysisEngine",
    "StaticAnalysisSummary",
    "StaticFinding",
]
