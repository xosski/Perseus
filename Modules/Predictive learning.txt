#!/usr/bin/env python3
"""
Predictive Learning Memory Layer for an LLM

Purpose:
- Track who did something
- When it happened
- What happened
- Why it likely happened
- What outcome followed
- What lesson the AI should remember
- Retrieve relevant lessons before future responses

This does NOT modify model weights.
It creates an external memory/reasoning system that can be attached to any LLM.
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional


DB_PATH = "predictive_learning_memory.db"


class PredictiveLearningMemory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                who TEXT,
                when_utc TEXT,
                what TEXT,
                why TEXT,
                context TEXT,
                outcome TEXT,
                lesson TEXT,
                confidence REAL,
                tags TEXT
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_name TEXT,
                description TEXT,
                evidence TEXT,
                prediction TEXT,
                confidence REAL,
                created_utc TEXT,
                updated_utc TEXT,
                tags TEXT
            )
            """)

            conn.commit()

    def _make_id(self, payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def add_event(
        self,
        who: str,
        what: str,
        why: str,
        context: str = "",
        outcome: str = "",
        lesson: str = "",
        confidence: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        when_utc = datetime.now(timezone.utc).isoformat()
        tags = tags or []

        raw = f"{who}|{when_utc}|{what}|{why}|{context}|{outcome}"
        event_id = self._make_id(raw)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO events
            (id, who, when_utc, what, why, context, outcome, lesson, confidence, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                who,
                when_utc,
                what,
                why,
                context,
                outcome,
                lesson,
                confidence,
                json.dumps(tags)
            ))
            conn.commit()

        return event_id

    def search_events(self, query: str, limit: int = 8) -> List[Dict]:
        """
        Simple keyword search.
        This can later be replaced with vector embeddings.
        """
        terms = [term.lower() for term in query.split() if term.strip()]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM events ORDER BY when_utc DESC")
            rows = cur.fetchall()

        scored = []
        for row in rows:
            text = " ".join([
                row["who"] or "",
                row["what"] or "",
                row["why"] or "",
                row["context"] or "",
                row["outcome"] or "",
                row["lesson"] or "",
                row["tags"] or ""
            ]).lower()

            score = sum(1 for term in terms if term in text)

            if score > 0:
                item = dict(row)
                item["score"] = score
                item["tags"] = json.loads(item["tags"] or "[]")
                scored.append(item)

        scored.sort(key=lambda x: (x["score"], x["when_utc"]), reverse=True)
        return scored[:limit]

    def add_pattern(
        self,
        pattern_name: str,
        description: str,
        evidence: str,
        prediction: str,
        confidence: float,
        tags: Optional[List[str]] = None
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        tags = tags or []

        raw = f"{pattern_name}|{description}|{prediction}"
        pattern_id = self._make_id(raw)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO patterns
            (id, pattern_name, description, evidence, prediction, confidence, created_utc, updated_utc, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id,
                pattern_name,
                description,
                evidence,
                prediction,
                confidence,
                now,
                now,
                json.dumps(tags)
            ))
            conn.commit()

        return pattern_id

    def predict_from_context(self, current_context: str, limit: int = 5) -> Dict:
        """
        Finds relevant past events and creates a prediction packet.
        The LLM can use this packet before answering.
        """
        related_events = self.search_events(current_context, limit=limit)

        if not related_events:
            return {
                "prediction": "No strong prior pattern found.",
                "confidence": 0.1,
                "relevant_lessons": [],
                "suggested_reasoning": "Ask clarifying questions and avoid assuming intent."
            }

        lessons = []
        total_confidence = 0

        for event in related_events:
            if event.get("lesson"):
                lessons.append(event["lesson"])
            total_confidence += float(event.get("confidence") or 0)

        avg_confidence = total_confidence / max(len(related_events), 1)

        return {
            "prediction": "Relevant prior events suggest this situation may follow a known pattern.",
            "confidence": round(avg_confidence, 2),
            "relevant_lessons": lessons,
            "related_events": related_events,
            "suggested_reasoning": (
                "Compare the current situation against prior lessons. "
                "Explain what is known, what is inferred, and what remains uncertain."
            )
        }

    def build_llm_context(self, user_message: str) -> str:
        """
        This produces a memory block you can prepend to an LLM prompt.
        """
        packet = self.predict_from_context(user_message)

        memory_lines = [
            "PREDICTIVE MEMORY CONTEXT",
            "Use this as background memory, not as absolute truth.",
            f"Prediction: {packet['prediction']}",
            f"Confidence: {packet['confidence']}",
            "",
            "Relevant lessons:"
        ]

        for lesson in packet.get("relevant_lessons", []):
            memory_lines.append(f"- {lesson}")

        memory_lines.append("")
        memory_lines.append(f"Current user message: {user_message}")

        return "\n".join(memory_lines)


if __name__ == "__main__":
    memory = PredictiveLearningMemory()

    # Example learning event
    memory.add_event(
        who="user",
        what="Asked about suspicious logs where standard login failed but AD authentication succeeded.",
        why="User was trying to understand whether authentication fallback could indicate tampering, misconfiguration, or identity bypass.",
        context="Forensic log analysis, authentication chain, DFS failure, Active Directory success.",
        outcome="Pattern suggested fallback authentication may be legitimate, but becomes suspicious when paired with hidden payouts or missing audit trails.",
        lesson=(
            "When primary authentication fails but AD succeeds, do not assume compromise by itself. "
            "Correlate with timing, privilege level, payout events, module control changes, and missing logs."
        ),
        confidence=0.82,
        tags=["forensics", "authentication", "active_directory", "logs"]
    )

    # Example prediction
    user_input = "A user fails normal login but validates through AD right before a payout event."
    context_block = memory.build_llm_context(user_input)

    print(context_block)