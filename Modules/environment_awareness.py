"""Bounded, read-only environment awareness for Perseus active cognition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import List, Optional


@dataclass(frozen=True)
class EnvironmentSnapshot:
    observed_at: str
    local_datetime: str
    timezone_name: str
    utc_offset: str
    working_directory: str
    project_name: str
    operating_system: str
    python_version: str
    git_branch: str
    git_changes: List[str]
    top_level_entries: List[str]

    def as_dict(self) -> dict:
        return {
            "observed_at": self.observed_at,
            "local_datetime": self.local_datetime,
            "timezone_name": self.timezone_name,
            "utc_offset": self.utc_offset,
            "working_directory": self.working_directory,
            "project_name": self.project_name,
            "operating_system": self.operating_system,
            "python_version": self.python_version,
            "git_branch": self.git_branch,
            "git_changes": list(self.git_changes),
            "top_level_entries": list(self.top_level_entries),
        }


class EnvironmentObserver:
    """Observes project/runtime facts without reading file contents or executing actions."""

    def __init__(self, root: Optional[str] = None, refresh_seconds: float = 5.0):
        self.root = Path(root or Path(__file__).resolve().parent.parent).resolve()
        self.refresh_seconds = max(0.0, float(refresh_seconds))
        self._lock = threading.RLock()
        self._cached: Optional[EnvironmentSnapshot] = None
        self._cached_at = 0.0

    def _git(self, *args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=str(self.root), capture_output=True,
                text=True, timeout=2, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return completed.stdout.strip() if completed.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def observe(self, force: bool = False) -> EnvironmentSnapshot:
        with self._lock:
            now = time.monotonic()
            if self._cached and not force and now - self._cached_at < self.refresh_seconds:
                return self._cached

            try:
                entries = sorted(
                    item.name + ("/" if item.is_dir() else "")
                    for item in self.root.iterdir()
                    if not item.name.startswith(".") and item.name != "__pycache__"
                )[:40]
            except OSError:
                entries = []

            status_lines = self._git("status", "--short").splitlines()[:25]
            observed_utc = datetime.now(timezone.utc)
            observed_local = observed_utc.astimezone()
            offset = observed_local.strftime("%z")
            offset_label = f"UTC{offset[:3]}:{offset[3:]}" if len(offset) == 5 else "unknown"
            self._cached = EnvironmentSnapshot(
                observed_at=observed_utc.isoformat(),
                local_datetime=observed_local.strftime("%A, %B %d, %Y at %I:%M:%S %p").replace(" 0", " "),
                timezone_name=observed_local.tzname() or "local time",
                utc_offset=offset_label,
                working_directory=str(self.root),
                project_name=self.root.name,
                operating_system=f"{platform.system()} {platform.release()} ({platform.machine()})",
                python_version=platform.python_version(),
                git_branch=self._git("branch", "--show-current") or "unknown",
                git_changes=status_lines,
                top_level_entries=entries,
            )
            self._cached_at = now
            return self._cached

    def build_prompt_context(self, _prompt: str = "") -> str:
        snapshot = self.observe()
        changes = ", ".join(snapshot.git_changes) or "clean"
        entries = ", ".join(snapshot.top_level_entries)
        return (
            "Read-only local environment snapshot (facts may become stale):\n"
            f"- Local date and time: {snapshot.local_datetime} ({snapshot.timezone_name}, {snapshot.utc_offset})\n"
            f"- Observed in UTC: {snapshot.observed_at}\n"
            f"- Project: {snapshot.project_name}\n"
            f"- Working directory: {snapshot.working_directory}\n"
            f"- Runtime: Python {snapshot.python_version} on {snapshot.operating_system}\n"
            f"- Git branch: {snapshot.git_branch}; changes: {changes}\n"
            f"- Top-level entries: {entries}\n"
            "Use this only to ground the current task. Do not infer permission to modify, "
            "delete, execute, network, or disclose anything."
        )


class Module(EnvironmentObserver):
    """Dynamic-module entry point."""
