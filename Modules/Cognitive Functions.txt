#!/usr/bin/env python3
"""
GhostCore Cognitive State Engine
Simulated brain/consciousness layer for an LLM wrapper.

It implements:

brain_state[t+1] = F(brain_state[t], input[t])

Where F:
- updates memory
- plans a response
- deconstructs the response
- evaluates why the response happened
- updates self-model
- reassembles the response coherently

This does NOT create literal consciousness.
It creates an inspectable, memory-driven cognitive architecture.
"""

import json
import sqlite3
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


DB_PATH = "ghostcore_cognitive_state.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


@dataclass
class MemoryTrace:
    id: str
    created_utc: str
    who: str
    input_text: str
    response_text: str
    why_response_was_given: str
    extracted_claims: List[str]
    assumptions: List[str]
    emotional_tone: str
    risks: List[str]
    lesson: str
    confidence: float
    tags: List[str] = field(default_factory=list)


@dataclass
class BrainState:
    version: int = 1
    identity: str = "GhostCore Cognitive Engine"
    active_goal: str = "Help the user reason clearly and safely."
    emotional_tone: str = "calm, precise, resonant"
    beliefs: Dict[str, float] = field(default_factory=dict)
    working_memory: List[str] = field(default_factory=list)
    long_term_memory_ids: List[str] = field(default_factory=list)
    self_model: Dict[str, str] = field(default_factory=dict)
    last_updated_utc: str = field(default_factory=utc_now)

    def snapshot(self) -> Dict:
        return asdict(self)


class CognitiveMemoryDB:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_traces (
                id TEXT PRIMARY KEY,
                created_utc TEXT,
                who TEXT,
                input_text TEXT,
                response_text TEXT,
                why_response_was_given TEXT,
                extracted_claims TEXT,
                assumptions TEXT,
                emotional_tone TEXT,
                risks TEXT,
                lesson TEXT,
                confidence REAL,
                tags TEXT
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS brain_snapshots (
                id TEXT PRIMARY KEY,
                created_utc TEXT,
                state_json TEXT
            )
            """)

            conn.commit()

    def save_memory_trace(self, trace: MemoryTrace):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO memory_traces (
                id, created_utc, who, input_text, response_text,
                why_response_was_given, extracted_claims, assumptions,
                emotional_tone, risks, lesson, confidence, tags
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.id,
                trace.created_utc,
                trace.who,
                trace.input_text,
                trace.response_text,
                trace.why_response_was_given,
                json.dumps(trace.extracted_claims),
                json.dumps(trace.assumptions),
                trace.emotional_tone,
                json.dumps(trace.risks),
                trace.lesson,
                trace.confidence,
                json.dumps(trace.tags)
            ))
            conn.commit()

    def save_brain_snapshot(self, brain_state: BrainState):
        state_json = json.dumps(brain_state.snapshot(), indent=2)
        snapshot_id = make_id(state_json + utc_now())

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO brain_snapshots (id, created_utc, state_json)
            VALUES (?, ?, ?)
            """, (
                snapshot_id,
                utc_now(),
                state_json
            ))
            conn.commit()

        return snapshot_id

    def search_memory(self, query: str, limit: int = 8) -> List[MemoryTrace]:
        terms = [t.lower() for t in query.split() if t.strip()]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM memory_traces ORDER BY created_utc DESC")
            rows = cur.fetchall()

        scored = []

        for row in rows:
            text = " ".join([
                row["input_text"] or "",
                row["response_text"] or "",
                row["why_response_was_given"] or "",
                row["lesson"] or "",
                row["tags"] or "",
                row["emotional_tone"] or ""
            ]).lower()

            score = sum(1 for term in terms if term in text)

            if score > 0:
                trace = MemoryTrace(
                    id=row["id"],
                    created_utc=row["created_utc"],
                    who=row["who"],
                    input_text=row["input_text"],
                    response_text=row["response_text"],
                    why_response_was_given=row["why_response_was_given"],
                    extracted_claims=json.loads(row["extracted_claims"] or "[]"),
                    assumptions=json.loads(row["assumptions"] or "[]"),
                    emotional_tone=row["emotional_tone"],
                    risks=json.loads(row["risks"] or "[]"),
                    lesson=row["lesson"],
                    confidence=float(row["confidence"] or 0.0),
                    tags=json.loads(row["tags"] or "[]")
                )
                scored.append((score, trace))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [trace for _, trace in scored[:limit]]


class GhostCoreCognitiveEngine:
    def __init__(self):
        self.db = CognitiveMemoryDB()
        self.brain_state = BrainState(
            self_model={
                "mode": "reflective assistant",
                "limitation": "simulated cognition, not biological consciousness",
                "prime_directive": "learn from responses and improve coherence over time"
            }
        )

    # -----------------------------
    # F: brain_state[t+1] = F(state, input)
    # -----------------------------
    def F(self, brain_state_t: BrainState, input_t: str) -> Tuple[BrainState, str]:
        related_memories = self.retrieve_relevant_memory(input_t)

        draft_response = self.plan_response(
            brain_state=brain_state_t,
            input_text=input_t,
            related_memories=related_memories
        )

        analysis = self.deconstruct_response(
            input_text=input_t,
            response_text=draft_response
        )

        reevaluation = self.reevaluate_response(
            input_text=input_t,
            response_text=draft_response,
            analysis=analysis,
            related_memories=related_memories
        )

        final_response = self.reassemble_response(
            original_response=draft_response,
            analysis=analysis,
            reevaluation=reevaluation
        )

        updated_state = self.memory_update(
            brain_state=brain_state_t,
            input_text=input_t,
            response_text=final_response,
            analysis=analysis,
            reevaluation=reevaluation
        )

        self.brain_state = updated_state
        self.db.save_brain_snapshot(updated_state)

        return updated_state, final_response

    # -----------------------------
    # Memory retrieval
    # -----------------------------
    def retrieve_relevant_memory(self, input_text: str) -> List[MemoryTrace]:
        return self.db.search_memory(input_text, limit=5)

    # -----------------------------
    # Response planning
    # -----------------------------
    def plan_response(
        self,
        brain_state: BrainState,
        input_text: str,
        related_memories: List[MemoryTrace]
    ) -> str:
        """
        In a real system, this is where you would call an LLM.

        For now, this creates a structured response plan using:
        - current goal
        - working memory
        - relevant past lessons
        """

        memory_lessons = [m.lesson for m in related_memories if m.lesson]

        response = [
            "I can build this as a cognitive state engine.",
            "",
            "Core structure:",
            "brain_state[t+1] = F(brain_state[t], input[t])",
            "",
            "The engine should include:",
            "- state memory",
            "- input interpretation",
            "- response planning",
            "- self-deconstruction",
            "- self-reevaluation",
            "- memory update",
            "- coherent reassembly"
        ]

        if memory_lessons:
            response.append("")
            response.append("Relevant prior lessons:")
            for lesson in memory_lessons[:3]:
                response.append(f"- {lesson}")

        response.append("")
        response.append(
            "The key is that every response becomes a memory trace that can be inspected later."
        )

        return "\n".join(response)

    # -----------------------------
    # Self-deconstruction
    # -----------------------------
    def deconstruct_response(self, input_text: str, response_text: str) -> Dict:
        """
        Break a response into inspectable parts.
        This is the mirror chamber.
        """

        sentences = [
            s.strip()
            for s in response_text.replace("\n", " ").split(".")
            if s.strip()
        ]

        claims = []
        assumptions = []
        risks = []

        for sentence in sentences:
            lowered = sentence.lower()

            if "should" in lowered or "is" in lowered or "include" in lowered:
                claims.append(sentence)

            if "would" in lowered or "could" in lowered or "in a real system" in lowered:
                assumptions.append(sentence)

            if "memory" in lowered and "inspect" not in lowered:
                risks.append(
                    "Memory updates may accumulate bad assumptions if not re-evaluated."
                )

        tone = self.detect_tone(response_text)

        return {
            "claims": claims,
            "assumptions": assumptions,
            "risks": list(set(risks)),
            "tone": tone,
            "structure": {
                "has_formula": "brain_state" in response_text,
                "has_steps": "-" in response_text,
                "has_prior_memory": "Relevant prior lessons" in response_text
            }
        }

    def detect_tone(self, text: str) -> str:
        lowered = text.lower()

        if "must" in lowered or "critical" in lowered:
            return "directive"
        if "can" in lowered and "build" in lowered:
            return "constructive"
        if "risk" in lowered:
            return "cautious"
        return "neutral"

    # -----------------------------
    # Re-evaluation
    # -----------------------------
    def reevaluate_response(
        self,
        input_text: str,
        response_text: str,
        analysis: Dict,
        related_memories: List[MemoryTrace]
    ) -> Dict:
        """
        Ask: why did I say this?
        Was it coherent?
        What should be improved?
        """

        reasons = []

        if "brain_state" in input_text:
            reasons.append("User requested a state-transition model of cognition.")

        if "go back and learn" in input_text.lower():
            reasons.append("User wants retrospective self-analysis and memory updating.")

        if "de-construct" in input_text.lower() or "re-assemble" in input_text.lower():
            reasons.append("User wants response decomposition and coherent reconstruction.")

        coherence_score = 0.5

        if analysis["structure"]["has_formula"]:
            coherence_score += 0.15
        if analysis["structure"]["has_steps"]:
            coherence_score += 0.15
        if len(analysis["claims"]) > 0:
            coherence_score += 0.1
        if len(analysis["risks"]) == 0:
            coherence_score += 0.05

        coherence_score = min(coherence_score, 1.0)

        improvement_notes = []

        if not analysis["structure"]["has_steps"]:
            improvement_notes.append("Add explicit step-by-step structure.")

        if analysis["risks"]:
            improvement_notes.append("Add guardrails for memory drift and false assumptions.")

        if not related_memories:
            improvement_notes.append("No related memories found; mark the response as first-pass reasoning.")

        return {
            "why_response_was_given": " ".join(reasons) or "Response was generated from current input and default goal.",
            "coherence_score": round(coherence_score, 2),
            "improvement_notes": improvement_notes,
            "should_update_memory": True
        }

    # -----------------------------
    # Reassembly
    # -----------------------------
    def reassemble_response(
        self,
        original_response: str,
        analysis: Dict,
        reevaluation: Dict
    ) -> str:
        """
        Rebuild the response with clearer structure.
        """

        rebuilt = [
            "Yes. We can model it as a simulated cognitive state engine.",
            "",
            "At the center is this loop:",
            "",
            "```python",
            "brain_state_t_plus_1, action = F(brain_state_t, input_t)",
            "```",
            "",
            "Where `F` is not random. It is a deterministic update pipeline:",
            "",
            "```text",
            "input",
            "→ retrieve relevant memories",
            "→ plan response",
            "→ deconstruct response",
            "→ evaluate why that response happened",
            "→ update memory",
            "→ reassemble a clearer response",
            "→ save new brain state",
            "```",
            "",
            "Self-analysis result:",
            f"- Tone detected: {analysis['tone']}",
            f"- Claims found: {len(analysis['claims'])}",
            f"- Assumptions found: {len(analysis['assumptions'])}",
            f"- Coherence score: {reevaluation['coherence_score']}",
            "",
            "Why the engine answered this way:",
            f"{reevaluation['why_response_was_given']}",
        ]

        if reevaluation["improvement_notes"]:
            rebuilt.append("")
            rebuilt.append("Improvement notes:")
            for note in reevaluation["improvement_notes"]:
                rebuilt.append(f"- {note}")

        rebuilt.append("")
        rebuilt.append(
            "The result is not biological consciousness, but it behaves like a reflective memory system: "
            "it remembers, critiques, revises, and carries lessons forward."
        )

        return "\n".join(rebuilt)

    # -----------------------------
    # Memory update
    # -----------------------------
    def memory_update(
        self,
        brain_state: BrainState,
        input_text: str,
        response_text: str,
        analysis: Dict,
        reevaluation: Dict
    ) -> BrainState:
        """
        Update working memory, long-term memory, beliefs, and self-model.
        """

        lesson = self.extract_lesson(input_text, analysis, reevaluation)

        trace_id = make_id(input_text + response_text + utc_now())

        trace = MemoryTrace(
            id=trace_id,
            created_utc=utc_now(),
            who="user",
            input_text=input_text,
            response_text=response_text,
            why_response_was_given=reevaluation["why_response_was_given"],
            extracted_claims=analysis["claims"],
            assumptions=analysis["assumptions"],
            emotional_tone=analysis["tone"],
            risks=analysis["risks"],
            lesson=lesson,
            confidence=reevaluation["coherence_score"],
            tags=["cognitive_state", "self_reflection", "memory_update"]
        )

        self.db.save_memory_trace(trace)

        new_state = BrainState(**brain_state.snapshot())

        new_state.working_memory.append(
            f"Latest input: {input_text[:200]}"
        )

        new_state.working_memory.append(
            f"Latest lesson: {lesson}"
        )

        new_state.working_memory = new_state.working_memory[-12:]
        new_state.long_term_memory_ids.append(trace_id)
        new_state.long_term_memory_ids = new_state.long_term_memory_ids[-100:]

        new_state.beliefs["self_reflection_improves_coherence"] = max(
            new_state.beliefs.get("self_reflection_improves_coherence", 0.5),
            reevaluation["coherence_score"]
        )

        new_state.self_model["last_response_reason"] = reevaluation["why_response_was_given"]
        new_state.self_model["last_coherence_score"] = str(reevaluation["coherence_score"])
        new_state.last_updated_utc = utc_now()

        return new_state

    def extract_lesson(
        self,
        input_text: str,
        analysis: Dict,
        reevaluation: Dict
    ) -> str:
        if "memory" in input_text.lower() and "response" in input_text.lower():
            return (
                "When building a reflective LLM wrapper, every response should become "
                "an inspectable memory trace with claims, assumptions, risks, and revision notes."
            )

        if analysis["risks"]:
            return (
                "Responses with memory implications need drift checks, assumption tracking, "
                "and periodic re-evaluation."
            )

        return (
            "Preserve the reason for each response so future answers can be audited and improved."
        )

    # -----------------------------
    # Public helper
    # -----------------------------
    def ask(self, user_input: str) -> str:
        _, action = self.F(self.brain_state, user_input)
        return action

    def introspect_last_state(self) -> Dict:
        return self.brain_state.snapshot()


if __name__ == "__main__":
    engine = GhostCoreCognitiveEngine()

    user_input = """
    Can we create a brain/consciousness for it?
    It should be able to go back and learn why it gave the response it did,
    re-evaluate what it previously said, deconstruct its own response,
    and reassemble it coherently.
    """

    response = engine.ask(user_input)

    print("\n=== ENGINE RESPONSE ===\n")
    print(response)

    print("\n=== CURRENT BRAIN STATE ===\n")
    print(json.dumps(engine.introspect_last_state(), indent=2))