"""Evidence-based response rationale storage; never private chain-of-thought."""
from __future__ import annotations
import hashlib, json, sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

class ResponseRationale:
    def __init__(self, db_path="perseus_response_rationale.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS response_rationales (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL,
                prompt_hash TEXT NOT NULL, response_hash TEXT NOT NULL,
                rationale_json TEXT NOT NULL)""")

    def record(self, prompt, response, provider, model, quality_score,
               quality_reasons=None, context_channels=None, dynamic_modules=None,
               intent="", strategy="", refined=False, fallback_used=False,
               introspection_changed=False):
        channels = list(dict.fromkeys(context_channels or []))
        modules = list(dict.fromkeys(dynamic_modules or []))
        factors = [
            f"The request was classified as {intent}." if intent else "The request was classified before generation.",
            f"The {provider} provider produced the selected response using {model}.",
            ("Grounding was contributed by: " + ", ".join(channels) + ".") if channels
            else "No retrieved knowledge or memory channel contributed context.",
        ]
        if strategy: factors.append(f"The deterministic response strategy was {strategy}.")
        if refined: factors.append("The initial draft was refined or repaired before selection.")
        if fallback_used: factors.append("A fallback path was selected because the preferred path was unavailable or weaker.")
        if introspection_changed: factors.append("Post-response critique changed the draft before it was returned.")
        rationale = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "explanation_type": "observable_pipeline_evidence",
            "limitation": "This explains recorded inputs and selection decisions, not private chain-of-thought or a complete account of model internals.",
            "intent": intent, "strategy": strategy, "provider": provider, "model": model,
            "quality_score": int(quality_score),
            "quality_reasons": [str(x) for x in (quality_reasons or []) if x][:6],
            "context_channels": channels, "dynamic_modules_used": modules,
            "refined": bool(refined), "fallback_used": bool(fallback_used),
            "introspection_changed": bool(introspection_changed), "selection_factors": factors,
        }
        digest = lambda text: hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            cursor = conn.execute("INSERT INTO response_rationales (created_utc,prompt_hash,response_hash,rationale_json) VALUES (?,?,?,?)",
                                  (rationale["created_utc"], digest(prompt), digest(response), json.dumps(rationale, ensure_ascii=False)))
            rationale["id"] = cursor.lastrowid
        return rationale

    def latest(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            row = conn.execute("SELECT id,rationale_json FROM response_rationales ORDER BY id DESC LIMIT 1").fetchone()
        if not row: return None
        result = json.loads(row[1]); result["id"] = row[0]
        return result

    def recent(self, limit=20):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            rows = conn.execute("SELECT id,rationale_json FROM response_rationales ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 200)),)).fetchall()
        results = []
        for row_id, payload in rows:
            item = json.loads(payload); item["id"] = row_id; results.append(item)
        return results
