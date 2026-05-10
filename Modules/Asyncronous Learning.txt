#!/usr/bin/env python3
"""
GhostCore Predictive Learning + AMM EchoWiring Memory Layer

Purpose:
- Store predictive learning events:
  who / when / why / what / outcome / lesson
- Add Asynchronous Memory Mapping fields:
  audio cue / rhythm / emotional tone / recall phrase / consent / safeguards
- Retrieve relevant memories and build an LLM-ready context block

Important:
This does not modify LLM weights.
It creates an external memory layer that an LLM can consult.
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional


DB_PATH = "ghostcore_echowiring_memory.db"


class EchoWiringMemory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS learning_events (
                id TEXT PRIMARY KEY,
                who TEXT,
                when_utc TEXT,
                what TEXT,
                why TEXT,
                context TEXT,
                outcome TEXT,
                lesson TEXT,
                confidence REAL,
                tags TEXT,

                -- AMM / EchoWiring fields
                amm_enabled INTEGER,
                consent_confirmed INTEGER,
                audio_cue TEXT,
                rhythm_pattern TEXT,
                emotional_tone TEXT,
                recall_phrase TEXT,
                stress_context TEXT,
                safety_notes TEXT
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
        tags: Optional[List[str]] = None,

        # EchoWiring / AMM fields
        amm_enabled: bool = False,
        consent_confirmed: bool = False,
        audio_cue: str = "",
        rhythm_pattern: str = "",
        emotional_tone: str = "",
        recall_phrase: str = "",
        stress_context: str = "",
        safety_notes: str = ""
    ) -> str:
        """
        Adds a memory event.

        AMM is only allowed when consent_confirmed=True.
        This keeps the system aligned with the EchoWiring ethical rule:
        no covert memory shaping.
        """

        if amm_enabled and not consent_confirmed:
            raise ValueError(
                "AMM/EchoWiring cannot be enabled without confirmed consent."
            )

        when_utc = datetime.now(timezone.utc).isoformat()
        tags = tags or []

        raw = f"{who}|{when_utc}|{what}|{why}|{context}|{outcome}|{audio_cue}|{recall_phrase}"
        event_id = self._make_id(raw)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO learning_events (
                id, who, when_utc, what, why, context, outcome, lesson,
                confidence, tags,
                amm_enabled, consent_confirmed, audio_cue, rhythm_pattern,
                emotional_tone, recall_phrase, stress_context, safety_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(tags),

                int(amm_enabled),
                int(consent_confirmed),
                audio_cue,
                rhythm_pattern,
                emotional_tone,
                recall_phrase,
                stress_context,
                safety_notes
            ))

            conn.commit()

        return event_id

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
            INSERT OR REPLACE INTO patterns (
                id, pattern_name, description, evidence, prediction,
                confidence, created_utc, updated_utc, tags
            )
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

    def search_events(self, query: str, limit: int = 8) -> List[Dict]:
        """
        Simple keyword search.
        Can later be replaced with vector embeddings.
        """

        terms = [term.lower() for term in query.split() if term.strip()]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM learning_events ORDER BY when_utc DESC")
            rows = cur.fetchall()

        scored = []

        for row in rows:
            item = dict(row)

            searchable_text = " ".join([
                item.get("who") or "",
                item.get("what") or "",
                item.get("why") or "",
                item.get("context") or "",
                item.get("outcome") or "",
                item.get("lesson") or "",
                item.get("tags") or "",
                item.get("audio_cue") or "",
                item.get("rhythm_pattern") or "",
                item.get("emotional_tone") or "",
                item.get("recall_phrase") or "",
                item.get("stress_context") or "",
                item.get("safety_notes") or ""
            ]).lower()

            score = sum(1 for term in terms if term in searchable_text)

            if score > 0:
                item["score"] = score
                item["tags"] = json.loads(item.get("tags") or "[]")
                scored.append(item)

        scored.sort(key=lambda x: (x["score"], x["when_utc"]), reverse=True)
        return scored[:limit]

    def predict_from_context(self, current_context: str, limit: int = 5) -> Dict:
        related_events = self.search_events(current_context, limit=limit)

        if not related_events:
            return {
                "prediction": "No strong prior memory pattern found.",
                "confidence": 0.1,
                "relevant_lessons": [],
                "amm_recall_cues": [],
                "suggested_reasoning": (
                    "Ask clarifying questions. Avoid assuming intent. "
                    "Separate known facts from inference."
                )
            }

        lessons = []
        recall_cues = []
        total_confidence = 0.0

        for event in related_events:
            total_confidence += float(event.get("confidence") or 0)

            if event.get("lesson"):
                lessons.append(event["lesson"])

            if event.get("amm_enabled"):
                recall_cues.append({
                    "audio_cue": event.get("audio_cue") or "",
                    "rhythm_pattern": event.get("rhythm_pattern") or "",
                    "emotional_tone": event.get("emotional_tone") or "",
                    "recall_phrase": event.get("recall_phrase") or "",
                    "stress_context": event.get("stress_context") or "",
                    "safety_notes": event.get("safety_notes") or ""
                })

        avg_confidence = total_confidence / max(len(related_events), 1)

        return {
            "prediction": "Relevant prior memory suggests this may follow a known pattern.",
            "confidence": round(avg_confidence, 2),
            "relevant_lessons": lessons,
            "amm_recall_cues": recall_cues,
            "related_events": related_events,
            "suggested_reasoning": (
                "Use prior lessons as context, not certainty. "
                "If AMM cues are present, use them as recall anchors, not commands."
            )
        }

    def build_llm_context(self, user_message: str) -> str:
        """
        Builds an LLM-ready memory context block.
        This can be prepended to a model prompt.
        """

        packet = self.predict_from_context(user_message)

        lines = [
            "GHOSTCORE PREDICTIVE MEMORY CONTEXT",
            "Use this as background memory, not absolute truth.",
            f"Prediction: {packet['prediction']}",
            f"Confidence: {packet['confidence']}",
            "",
            "Relevant lessons:"
        ]

        for lesson in packet.get("relevant_lessons", []):
            lines.append(f"- {lesson}")

        if packet.get("amm_recall_cues"):
            lines.append("")
            lines.append("EchoWiring / AMM recall cues:")
            for cue in packet["amm_recall_cues"]:
                lines.append(f"- Audio cue: {cue['audio_cue']}")
                lines.append(f"  Rhythm: {cue['rhythm_pattern']}")
                lines.append(f"  Emotional tone: {cue['emotional_tone']}")
                lines.append(f"  Recall phrase: {cue['recall_phrase']}")
                lines.append(f"  Stress context: {cue['stress_context']}")
                lines.append(f"  Safety notes: {cue['safety_notes']}")

        lines.append("")
        lines.append(f"Current user message: {user_message}")

        return "\n".join(lines)

    def export_json(self, output_path: str = "ghostcore_echowiring_export.json"):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("SELECT * FROM learning_events ORDER BY when_utc DESC")
            events = [dict(row) for row in cur.fetchall()]

            cur.execute("SELECT * FROM patterns ORDER BY updated_utc DESC")
            patterns = [dict(row) for row in cur.fetchall()]

        payload = {
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "events": events,
            "patterns": patterns
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return output_path


if __name__ == "__main__":
    memory = EchoWiringMemory()

    # Example 1: forensic learning event with AMM disabled
    memory.add_event(
        who="user",
        what="Observed login failure through default DFS authentication followed by AD success.",
        why="User wanted to determine whether fallback authentication was normal, suspicious, or tied to a hidden transaction.",
        context="Authentication logs, DFS failure, Active Directory success, payout correlation.",
        outcome="The pattern is not automatically malicious, but becomes suspicious when paired with missing payout records or privilege-sensitive events.",
        lesson=(
            "Authentication fallback is not proof of compromise alone. "
            "Correlate with timing, role, transaction events, device errors, and missing logs."
        ),
        confidence=0.84,
        tags=["forensics", "authentication", "active_directory", "logs"]
    )

    # Example 2: EchoWiring-enabled learning event
    memory.add_event(
        who="operator",
        what="Learned emergency oxygen loop calibration sequence.",
        why="Procedure must be recalled under panic, alarm noise, and cognitive overload.",
        context="Spacecraft emergency procedure training.",
        outcome="Recall improved when paired with a slow four-beat breathing rhythm.",
        lesson=(
            "For emergency procedures, pair each step with a stable rhythm and a short recall phrase."
        ),
        confidence=0.78,
        tags=["AMM", "EchoWiring", "training", "emergency_recall"],

        amm_enabled=True,
        consent_confirmed=True,
        audio_cue="low cello drone in E minor",
        rhythm_pattern="4-count inhale, 4-count hold, 4-count action",
        emotional_tone="calm urgency",
        recall_phrase="The reactor remembers the melody.",
        stress_context="alarm state, low oxygen, operator panic",
        safety_notes=(
            "Use only with consent. Do not use for covert conditioning. "
            "Avoid trauma-linked tones unless supervised."
        )
    )

    # Example query
    user_message = "How should the AI remember emergency procedures under stress?"
    print(memory.build_llm_context(user_message))

    # Optional export
    exported = memory.export_json()
    print(f"\nExported memory archive to: {exported}")