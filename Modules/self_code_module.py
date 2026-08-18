"""User-gated, workspace-confined staged code updates for Perseus."""
from __future__ import annotations
import ast, difflib, hashlib, json, os, re, shutil, tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_SUFFIXES = {".py", ".pyw", ".json", ".md", ".txt"}
MAX_UPDATE_CHARS = 500_000

@dataclass
class UpdateProposal:
    proposal_id: str
    relative_path: str
    source: str = field(repr=False)
    diff: str = ""
    rationale: str = ""
    created_utc: str = ""
    applied: bool = False

class SelfCodeModule:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.backup_root = self.project_root / ".perseus_self_code_backups"
        self._proposals = {}

    def build_prompt_context(self, prompt):
        lower = (prompt or "").lower()
        if not any(x in lower for x in ("self code", "self-code", "update yourself", "modify yourself")): return ""
        return ("SELF-CODE MODULE IS USER-ENABLED FOR THIS SESSION.\n"
                "Propose code, but never claim a file changed unless the host reports an approved apply.\n"
                "Every change must be staged, diffed, syntax-checked, and separately approved by the user.\n"
                "Never weaken approval gates, path confinement, safety controls, or auditability.")

    def status(self):
        return {"loaded": True, "project_root": str(self.project_root),
                "pending_proposals": sum(not x.applied for x in self._proposals.values()),
                "write_requires_explicit_approval": True}

    def stage_update(self, relative_path, source, rationale=""):
        target = self._resolve_target(relative_path); source = source or ""
        if len(source) > MAX_UPDATE_CHARS: raise ValueError(f"Update exceeds {MAX_UPDATE_CHARS} characters")
        self._validate_source(target, source)
        old = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        relative = target.relative_to(self.project_root).as_posix()
        diff = "".join(difflib.unified_diff(old.splitlines(keepends=True), source.splitlines(keepends=True),
                                             fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        if not diff: raise ValueError("Proposed content is identical to the current file")
        created = datetime.now(timezone.utc).isoformat()
        proposal_id = hashlib.sha256(f"{target}|{created}|{source}".encode()).hexdigest()[:16]
        proposal = UpdateProposal(proposal_id, relative, source, diff, (rationale or "")[:2000], created)
        self._proposals[proposal_id] = proposal
        return self._public(proposal)

    def apply_update(self, proposal_id, approved=False):
        if not approved: return {"ok": False, "error": "Explicit user approval is required; pass approved=True."}
        proposal = self._proposals.get(proposal_id)
        if not proposal: return {"ok": False, "error": "Unknown proposal ID."}
        if proposal.applied: return {"ok": False, "error": "Proposal was already applied."}
        target = self._resolve_target(proposal.relative_path); self._validate_source(target, proposal.source)
        backup = None
        if target.exists():
            backup = self.backup_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / proposal.relative_path
            backup.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(proposal.source); handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name): os.unlink(temp_name)
        proposal.applied = True
        return {"ok": True, "proposal_id": proposal_id, "path": str(target),
                "backup": str(backup) if backup else None,
                "restart_required": target.suffix.lower() in {".py", ".pyw"}}

    def get_proposal(self, proposal_id):
        item = self._proposals.get(proposal_id)
        return ({"ok": True, **self._public(item)} if item else {"ok": False, "error": "Unknown proposal ID."})

    def _resolve_target(self, relative_path):
        raw = (relative_path or "").strip().replace("\\", "/")
        if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw): raise ValueError("A workspace-relative path is required")
        target = (self.project_root / raw).resolve()
        try: parts = target.relative_to(self.project_root).parts
        except ValueError as exc: raise ValueError("Target must remain inside the Perseus workspace") from exc
        if any(x.startswith(".") for x in parts): raise ValueError("Hidden files and directories cannot be modified")
        if target.suffix.lower() not in ALLOWED_SUFFIXES: raise ValueError("Only Python, JSON, Markdown, and text files can be updated")
        return target

    @staticmethod
    def _validate_source(target, source):
        if target.suffix.lower() in {".py", ".pyw"}:
            try: ast.parse(source, filename=str(target))
            except SyntaxError as exc: raise ValueError(f"Python syntax error on line {exc.lineno}: {exc.msg}") from exc
        elif target.suffix.lower() == ".json":
            try: json.loads(source)
            except json.JSONDecodeError as exc: raise ValueError(f"JSON syntax error on line {exc.lineno}: {exc.msg}") from exc

    @staticmethod
    def _public(item):
        return {"proposal_id": item.proposal_id, "relative_path": item.relative_path, "diff": item.diff,
                "rationale": item.rationale, "created_utc": item.created_utc, "applied": item.applied,
                "requires_explicit_approval": True}
