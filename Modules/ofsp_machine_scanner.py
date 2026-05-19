"""
OFSP-Inspired Machine Scanner

Defensive host scan utilities adapted for HadesAI integration.
This module is read-only and triage-focused.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

try:
    from modules.endpoint_heuristics_detector import run_scan as run_endpoint_heuristics_scan
    HAS_ENDPOINT_HEURISTICS = True
except Exception:
    run_endpoint_heuristics_scan = None
    HAS_ENDPOINT_HEURISTICS = False


SUSPICIOUS_NAME_PATTERNS = [
    r"^svchost\d*\.exe$",
    r"^explorer\d*\.exe$",
    r"^lsass\d*\.exe$",
    r"^csrss\d*\.exe$",
    r"^[a-f0-9]{8,}\.(exe|dll|tmp|dat|bin)$",
]

SUSPICIOUS_CMD_PATTERNS = [
    r"powershell.*-enc(?:odedcommand)?",
    r"frombase64string",
    r"wscript\.shell",
    r"virtualalloc",
    r"writeprocessmemory",
    r"rundll32.*http",
    r"regsvr32.*scrobj\.dll",
]

SUSPICIOUS_PORTS = {4444, 1337, 5555, 6666, 8081, 8443, 9001, 31337}


def _file_sha256(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    sha = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception:
        return ""


def _severity_from_score(score: int) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def _collect_startup_items() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    startup_dirs = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup",
        Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup",
    ]

    for startup_dir in startup_dirs:
        if not startup_dir.exists() or not startup_dir.is_dir():
            continue
        for path in startup_dir.rglob("*"):
            if path.is_file():
                items.append({
                    "type": "startup_file",
                    "path": str(path),
                    "name": path.name,
                })

    # Optional registry startup keys on Windows.
    if platform.system().lower() == "windows":
        try:
            import winreg  # type: ignore

            reg_targets = [
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
            ]
            for hive, key_path in reg_targets:
                try:
                    with winreg.OpenKey(hive, key_path) as key:
                        index = 0
                        while True:
                            try:
                                name, value, _ = winreg.EnumValue(key, index)
                                items.append({
                                    "type": "startup_registry",
                                    "path": key_path,
                                    "name": name,
                                    "value": str(value),
                                })
                                index += 1
                            except OSError:
                                break
                except OSError:
                    continue
        except Exception:
            pass

    return items


def _scan_processes(
    include_hashes: bool = False,
    max_processes: Optional[int] = None,
    include_network: bool = True,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    net_by_pid: Dict[int, List[Dict[str, Any]]] = {}
    if include_network:
        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.pid is None:
                    continue
                net_by_pid.setdefault(conn.pid, []).append({
                    "status": str(conn.status),
                    "local": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                    "remote": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                    "remote_port": conn.raddr.port if conn.raddr else 0,
                })
        except Exception:
            pass

    proc_iter = list(psutil.process_iter(attrs=["pid", "name", "exe", "cmdline", "username"]))
    if isinstance(max_processes, int) and max_processes > 0:
        proc_iter = proc_iter[:max_processes]

    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue

        name = (info.get("name") or "").lower()
        exe = info.get("exe") or ""
        cmdline = " ".join(info.get("cmdline") or [])
        score = 0
        reasons: List[str] = []

        for pattern in SUSPICIOUS_NAME_PATTERNS:
            if re.search(pattern, name, re.IGNORECASE):
                score += 15
                reasons.append("Suspicious process naming pattern")
                break

        for pattern in SUSPICIOUS_CMD_PATTERNS:
            if re.search(pattern, cmdline, re.IGNORECASE):
                score += 25
                reasons.append("Suspicious command-line execution pattern")
                break

        if exe:
            exe_lower = exe.lower()
            if "\\appdata\\" in exe_lower or "\\temp\\" in exe_lower:
                score += 20
                reasons.append("Process executing from user-writable path")

        connections = net_by_pid.get(int(info.get("pid") or 0), [])
        for conn in connections:
            remote_port = int(conn.get("remote_port") or 0)
            if remote_port in SUSPICIOUS_PORTS:
                score += 20
                reasons.append(f"Connected to suspicious remote port {remote_port}")

        if score <= 0:
            continue

        findings.append({
            "source": "ofsp_process_scan",
            "pid": int(info.get("pid") or 0),
            "process": name or "unknown",
            "exe": exe,
            "cmdline": cmdline,
            "username": info.get("username") or "",
            "score": score,
            "severity": _severity_from_score(score),
            "reasons": reasons,
            "network": connections,
            "sha256": _file_sha256(exe) if (include_hashes and exe) else "",
        })

    findings.sort(key=lambda item: (-int(item.get("score", 0)), item.get("process", "")))
    return findings


def run_machine_scan(
    deep: bool = False,
    ofsp_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """Run OFSP-inspired machine scan and return structured defensive findings."""
    progress = progress_callback or (lambda _: None)
    progress(5)

    findings: List[Dict[str, Any]] = []
    startup_items: List[Dict[str, str]] = []

    include_hashes = bool(deep)
    max_processes = 600 if deep else 220
    include_network = bool(deep)

    process_findings = _scan_processes(
        include_hashes=include_hashes,
        max_processes=max_processes,
        include_network=include_network,
    )
    findings.extend(process_findings)
    progress(45)

    endpoint_heuristics: Dict[str, Any] = {}
    if deep and HAS_ENDPOINT_HEURISTICS and run_endpoint_heuristics_scan:
        try:
            def _heur_progress(pct: int):
                # Map endpoint heuristics internal progress into 45-70 range.
                progress(45 + int(max(0, min(100, pct)) * 0.25))

            endpoint_heuristics = run_endpoint_heuristics_scan(
                check_signatures=False,
                max_processes=max_processes,
                progress_callback=_heur_progress,
            )
            for item in endpoint_heuristics.get("findings", []):
                findings.append({
                    "source": "endpoint_heuristics",
                    "pid": item.get("pid", 0),
                    "process": item.get("name", "unknown"),
                    "exe": item.get("exe") or "",
                    "cmdline": item.get("cmdline") or "",
                    "username": item.get("username") or "",
                    "score": int(item.get("score", 0)),
                    "severity": item.get("severity", "LOW"),
                    "reasons": item.get("reasons", []),
                    "network": item.get("network", []),
                    "sha256": _file_sha256(item.get("exe") or "") if (include_hashes and item.get("exe")) else "",
                })
        except Exception:
            endpoint_heuristics = {"error": "endpoint heuristics unavailable"}
    else:
        progress(70)

    if deep:
        startup_items = _collect_startup_items()
        for item in startup_items:
            path = (item.get("path") or "").lower()
            score = 0
            reasons: List[str] = []
            if any(token in path for token in ["temp", "appdata", "downloads"]):
                score += 20
                reasons.append("Startup item references user-writable path")
            if item.get("type") == "startup_registry" and re.search(r"powershell|cmd\.exe|wscript|mshta", str(item.get("value", "")), re.IGNORECASE):
                score += 30
                reasons.append("Startup registry entry launches script/command interpreter")

            if score > 0:
                findings.append({
                    "source": "ofsp_startup_scan",
                    "pid": 0,
                    "process": item.get("name", "startup_item"),
                    "exe": item.get("path", ""),
                    "cmdline": item.get("value", ""),
                    "username": "",
                    "score": score,
                    "severity": _severity_from_score(score),
                    "reasons": reasons,
                    "network": [],
                    "sha256": _file_sha256(item.get("path", "")) if item.get("type") == "startup_file" else "",
                })
    progress(90)

    findings.sort(key=lambda item: (-int(item.get("score", 0)), item.get("process", ""), int(item.get("pid", 0))))

    ofsp_exists = bool(ofsp_path and os.path.isdir(ofsp_path))
    hostname = socket.gethostname()

    report = {
        "scan_type": "deep" if deep else "quick",
        "generated_at": datetime.now().isoformat(),
        "host": {
            "hostname": hostname,
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        },
        "ofsp_reference": {
            "path": ofsp_path or "",
            "available": ofsp_exists,
        },
        "summary": {
            "total_findings": len(findings),
            "high": sum(1 for item in findings if item.get("severity") == "HIGH"),
            "medium": sum(1 for item in findings if item.get("severity") == "MEDIUM"),
            "low": sum(1 for item in findings if item.get("severity") == "LOW"),
            "sources": {
                "ofsp_process_scan": sum(1 for item in findings if item.get("source") == "ofsp_process_scan"),
                "endpoint_heuristics": sum(1 for item in findings if item.get("source") == "endpoint_heuristics"),
                "ofsp_startup_scan": sum(1 for item in findings if item.get("source") == "ofsp_startup_scan"),
            },
            "startup_items_scanned": len(startup_items),
            "processes_seen": len(list(psutil.process_iter(attrs=[]))),
        },
        "endpoint_heuristics": endpoint_heuristics,
        "findings": findings,
    }
    progress(100)
    return report
