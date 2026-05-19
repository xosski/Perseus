"""
Endpoint Heuristics Detector

Defensive heuristic detection for Windows endpoints.
Scores suspicious processes and returns findings as structured data.

Notes:
- Heuristic scoring is not proof of compromise.
- This is meant for triage and hunting, not automatic blocking.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import psutil

USER_WRITABLE_PATTERNS = [
    r"\\Users\\[^\\]+\\AppData\\Local\\Temp\\",
    r"\\Users\\[^\\]+\\AppData\\Roaming\\",
    r"\\Users\\[^\\]+\\Downloads\\",
    r"\\Users\\Public\\",
    r"\\ProgramData\\",
    r"\\Temp\\",
]

LOLBINS = {
    "powershell.exe",
    "pwsh.exe",
    "cmd.exe",
    "wscript.exe",
    "cscript.exe",
    "mshta.exe",
    "rundll32.exe",
    "regsvr32.exe",
    "certutil.exe",
    "bitsadmin.exe",
    "wmic.exe",
    "msbuild.exe",
    "installutil.exe",
}

SUSPICIOUS_PARENTS = {
    "winword.exe",
    "excel.exe",
    "outlook.exe",
    "acrord32.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
}

SUSPICIOUS_EXTENSIONS = {".tmp", ".dat", ".bin", ".scr", ".js", ".jse", ".vbs", ".vbe", ".hta"}
SUSPICIOUS_PORTS = {4444, 1337, 5555, 6666, 8081, 8443, 9001, 31337}
HIGH_RISK_SCORE = 70
MEDIUM_RISK_SCORE = 40


@dataclass
class Finding:
    pid: int
    name: str
    score: int = 0
    severity: str = "LOW"
    exe: Optional[str] = None
    cmdline: Optional[str] = None
    parent_name: Optional[str] = None
    username: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    network: List[Dict[str, str]] = field(default_factory=list)


def normalize_path(path: Optional[str]) -> str:
    return (path or "").strip().lower()


def matches_any_pattern(path: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, path, re.IGNORECASE) for pattern in patterns)


def safe_join_cmdline(parts: List[str]) -> str:
    try:
        return " ".join(parts)
    except Exception:
        return ""


def is_encoded_powershell(cmdline: str) -> bool:
    patterns = [
        r"-enc(?:odedcommand)?\s+",
        r"frombase64string",
        r"\[system\.convert\]::frombase64string",
    ]
    return any(re.search(pattern, cmdline, re.IGNORECASE) for pattern in patterns)


def has_suspicious_switches(cmdline: str) -> bool:
    patterns = [
        r"-nop\b",
        r"-w\s+hidden",
        r"-windowstyle\s+hidden",
        r"-executionpolicy\s+bypass",
        r"-ep\s+bypass",
        r"-noni\b",
        r"/c\s+",
    ]
    return any(re.search(pattern, cmdline, re.IGNORECASE) for pattern in patterns)


def suspicious_filename(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(tuple(SUSPICIOUS_EXTENSIONS)):
        return True
    if re.match(r"^[a-f0-9]{8,}\.(exe|dll|tmp|dat|bin)$", lower):
        return True
    if "~" in lower:
        return True
    return False


def powershell_authenticode_status(path: str) -> str:
    """Use PowerShell to check Authenticode signature state."""
    if not path or not Path(path).exists():
        return "Missing"

    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "$p = %r; "
            "try { "
            "(Get-AuthenticodeSignature -FilePath $p).Status.ToString() "
            "} catch { 'UnknownError' }"
        ) % path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return (result.stdout or "").strip() or "UnknownError"
    except Exception:
        return "UnknownError"


def build_process_map() -> Dict[int, psutil.Process]:
    proc_map: Dict[int, psutil.Process] = {}
    for proc in psutil.process_iter(attrs=[]):
        try:
            proc_map[proc.pid] = proc
        except Exception:
            continue
    return proc_map


def collect_network_by_pid() -> Dict[int, List[Dict[str, str]]]:
    net_map: Dict[int, List[Dict[str, str]]] = {}
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.pid is None:
                continue
            local = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
            remote = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else ""
            entry = {
                "status": str(conn.status),
                "local": local,
                "remote": remote,
            }
            net_map.setdefault(conn.pid, []).append(entry)
    except Exception:
        pass
    return net_map


def score_process(
    proc: psutil.Process,
    proc_map: Dict[int, psutil.Process],
    net_map: Dict[int, List[Dict[str, str]]],
    check_signatures: bool = True,
    signature_cache: Optional[Dict[str, str]] = None,
) -> Optional[Finding]:
    try:
        info = proc.as_dict(attrs=["pid", "name", "exe", "cmdline", "ppid", "username"])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

    pid = info["pid"]
    name = (info.get("name") or "").lower()
    exe = info.get("exe") or ""
    cmdline = safe_join_cmdline(info.get("cmdline") or [])
    ppid = info.get("ppid") or 0
    username = info.get("username") or ""
    parent_name = ""

    if ppid in proc_map:
        try:
            parent_name = (proc_map[ppid].name() or "").lower()
        except Exception:
            parent_name = ""

    finding = Finding(
        pid=pid,
        name=name,
        exe=exe or None,
        cmdline=cmdline or None,
        parent_name=parent_name or None,
        username=username or None,
    )

    exe_lower = normalize_path(exe)
    cmd_lower = cmdline.lower()

    if exe_lower and matches_any_pattern(exe_lower, USER_WRITABLE_PATTERNS):
        finding.score += 20
        finding.reasons.append("Executable launched from user-writable or staging path")

    if suspicious_filename(name) or suspicious_filename(Path(exe_lower).name):
        finding.score += 15
        finding.reasons.append("Suspicious filename or extension")

    if name in LOLBINS:
        finding.score += 10
        finding.reasons.append("Living-off-the-land binary in use")

    if "powershell" in name or "pwsh" in name:
        if is_encoded_powershell(cmd_lower):
            finding.score += 30
            finding.reasons.append("Encoded PowerShell usage detected")
        if has_suspicious_switches(cmd_lower):
            finding.score += 20
            finding.reasons.append("PowerShell stealth or bypass switches detected")

    if re.search(r"https?://", cmd_lower) and name in LOLBINS:
        finding.score += 20
        finding.reasons.append("LOLBIN launched with URL or remote content reference")

    if parent_name in SUSPICIOUS_PARENTS and name in LOLBINS:
        finding.score += 25
        finding.reasons.append(f"Suspicious parent-child chain: {parent_name} -> {name}")

    if name in {"rundll32.exe", "regsvr32.exe"} and re.search(r"(http|scrobj\.dll|javascript:|vbscript:)", cmd_lower):
        finding.score += 35
        finding.reasons.append("Suspicious DLL/scriptlet execution pattern")

    if name == "certutil.exe" and re.search(r"(-urlcache|-decode|-encode|-split)", cmd_lower):
        finding.score += 25
        finding.reasons.append("Certutil used with download or transform switches")

    if check_signatures and exe and Path(exe).exists():
        sig_status = None
        if signature_cache is not None:
            sig_status = signature_cache.get(exe)

        if sig_status is None:
            sig_status = powershell_authenticode_status(exe)
            if signature_cache is not None:
                signature_cache[exe] = sig_status

        if sig_status not in {"Valid", "UnknownError"}:
            finding.score += 20
            finding.reasons.append(f"Executable signature status: {sig_status}")

    connections = net_map.get(pid, [])
    finding.network = connections

    for conn in connections:
        local_port = int(conn["local"].rsplit(":", 1)[1]) if ":" in conn["local"] else -1
        remote_port = int(conn["remote"].rsplit(":", 1)[1]) if ":" in conn["remote"] else -1

        if conn["status"].upper() == "LISTEN" and local_port in SUSPICIOUS_PORTS:
            finding.score += 15
            finding.reasons.append(f"Listening on suspicious port {local_port}")

        if remote_port in SUSPICIOUS_PORTS:
            finding.score += 20
            finding.reasons.append(f"Connected to suspicious remote port {remote_port}")

        if name in LOLBINS and conn["remote"]:
            finding.score += 15
            finding.reasons.append("LOLBIN process has active network connection")

    if exe_lower and "\\windows\\tasks\\" in exe_lower:
        finding.score += 15
        finding.reasons.append("Process executing from suspicious Windows Tasks path")

    if re.search(r"appdata\\.*\\(svchost|lsass|csrss|explorer)\.exe", exe_lower):
        finding.score += 35
        finding.reasons.append("System-like executable name running from user path")

    if finding.score >= HIGH_RISK_SCORE:
        finding.severity = "HIGH"
    elif finding.score >= MEDIUM_RISK_SCORE:
        finding.severity = "MEDIUM"
    else:
        finding.severity = "LOW"

    return finding if finding.score > 0 else None


def run_scan(
    check_signatures: bool = True,
    max_processes: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, object]:
    """Run endpoint heuristics scan and return structured findings."""
    proc_map = build_process_map()
    net_map = collect_network_by_pid()

    findings: List[Finding] = []
    proc_list = list(proc_map.values())
    if isinstance(max_processes, int) and max_processes > 0:
        proc_list = proc_list[:max_processes]

    total = len(proc_list) or 1
    callback = progress_callback or (lambda _: None)
    signature_cache: Dict[str, str] = {}
    signature_checks_enabled = bool(check_signatures and platform.system().lower() == "windows")

    for idx, proc in enumerate(proc_list):
        finding = score_process(
            proc,
            proc_map,
            net_map,
            check_signatures=signature_checks_enabled,
            signature_cache=signature_cache,
        )
        if finding:
            findings.append(finding)
        if idx % 10 == 0 or idx == total - 1:
            callback(int(((idx + 1) / total) * 100))

    findings.sort(key=lambda item: (-item.score, item.name, item.pid))

    return {
        "total_findings": len(findings),
        "high": sum(1 for item in findings if item.severity == "HIGH"),
        "medium": sum(1 for item in findings if item.severity == "MEDIUM"),
        "low": sum(1 for item in findings if item.severity == "LOW"),
        "findings": [asdict(item) for item in findings],
    }


def main() -> int:
    print(json.dumps(run_scan(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
