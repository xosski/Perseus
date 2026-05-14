#!/usr/bin/env python3
"""
Brain State Module for Perseus / PortableLLM

Implements a deterministic brain-state update loop:

    brain_state[t+1] = F(brain_state[t], input[t])

This is not biological consciousness and it does not alter model weights.
It is a persistent cognitive control layer that tracks attention, goals,
confidence, uncertainty, affect-like tone variables, topic focus, and lessons.

The function F is deterministic by default: no coin flips, no randomness.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


DEFAULT_DB_PATH = "brain_state_memory.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


@dataclass
class BrainState:
    """Compact persistent cognitive state."""

    version: int = 1
    updated_utc: str = field(default_factory=utc_now)

    # Control variables, 0.0 to 1.0
    attention: float = 0.55
    curiosity: float = 0.50
    caution: float = 0.45
    cognitive_load: float = 0.25
    confidence: float = 0.55
    uncertainty: float = 0.45
    social_warmth: float = 0.50
    directness: float = 0.65

    # Working memory
    active_topic: str = ""
    active_intent: str = "general"
    active_goal: str = "answer directly and usefully"
    focus_terms: List[str] = field(default_factory=list)
    recent_inputs: List[str] = field(default_factory=list)
    recent_actions: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)

    # Diagnostics
    update_count: int = 0


@dataclass
class BrainAction:
    """Action plan produced from the updated brain state."""

    intent: str
    goal: str
    response_strategy: str
    retrieval_strategy: str
    uncertainty_policy: str
    tone: str
    constraints: List[str]
    focus_terms: List[str]
    confidence: float
    reason: str


class BrainStateEngine:
    """
    Deterministic state machine.

    F(state_t, input_t) consists of:
    1. parse input signals
    2. update control variables
    3. update working memory
    4. plan a response action
    5. persist state and transition trace
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()
        self.brain_state = self.load_latest_state() or BrainState()
        self.save_state(self.brain_state, reason="engine_init")

    # -------------------------
    # Storage
    # -------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS brain_state_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_utc TEXT,
                state_json TEXT,
                reason TEXT
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS brain_transitions (
                id TEXT PRIMARY KEY,
                created_utc TEXT,
                input_text TEXT,
                input_signals_json TEXT,
                previous_state_json TEXT,
                updated_state_json TEXT,
                action_json TEXT,
                response_text TEXT,
                quality_score INTEGER,
                lesson TEXT
            )
            """)
            conn.commit()

    def load_latest_state(self) -> Optional[BrainState]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT state_json FROM brain_state_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()

        if not row:
            return None

        try:
            data = json.loads(row["state_json"])
            return BrainState(**data)
        except Exception:
            return None

    def save_state(self, state: BrainState, reason: str = "update") -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO brain_state_snapshots (created_utc, state_json, reason)
                VALUES (?, ?, ?)
                """,
                (utc_now(), json.dumps(asdict(state), ensure_ascii=False), reason),
            )
            conn.commit()

    def _save_transition(
        self,
        transition_id: str,
        input_text: str,
        input_signals: Dict,
        previous_state: BrainState,
        updated_state: BrainState,
        action: BrainAction,
        response_text: str = "",
        quality_score: Optional[int] = None,
        lesson: str = "",
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO brain_transitions (
                    id, created_utc, input_text, input_signals_json,
                    previous_state_json, updated_state_json, action_json,
                    response_text, quality_score, lesson
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition_id,
                    utc_now(),
                    input_text,
                    json.dumps(input_signals, ensure_ascii=False),
                    json.dumps(asdict(previous_state), ensure_ascii=False),
                    json.dumps(asdict(updated_state), ensure_ascii=False),
                    json.dumps(asdict(action), ensure_ascii=False),
                    response_text,
                    int(quality_score) if quality_score is not None else None,
                    lesson,
                ),
            )
            conn.commit()

    # -------------------------
    # Public API
    # -------------------------

    def F(self, brain_state_t: BrainState, input_t: str, profile: Optional[Dict] = None) -> Tuple[BrainState, BrainAction]:
        """Update brain based on past state and present input."""
        signals = self.parse_input(input_t, profile=profile)
        updated_state = self.memory_update(brain_state_t, input_t, signals)
        action = self.plan_response(updated_state, signals)
        return updated_state, action

    def step_input(self, input_text: str, profile: Optional[Dict] = None) -> Tuple[BrainState, BrainAction]:
        previous = self.brain_state
        updated, action = self.F(previous, input_text, profile=profile)
        self.brain_state = updated
        self.save_state(updated, reason="input_step")
        self._save_transition(
            transition_id=stable_id(f"{previous.update_count}|{input_text}|{utc_now()}"),
            input_text=input_text,
            input_signals=self.parse_input(input_text, profile=profile),
            previous_state=previous,
            updated_state=updated,
            action=action,
        )
        return updated, action

    def update_after_response(
        self,
        input_text: str,
        response_text: str,
        quality_score: int = 70,
        issues: Optional[List[str]] = None,
    ) -> BrainState:
        """
        Learn from the produced response without changing model weights.
        This adjusts the control state and stores a lesson.
        """
        issues = issues or []
        state = self.brain_state
        q = clamp(float(quality_score) / 100.0)

        state.confidence = clamp((state.confidence * 0.75) + (q * 0.25))
        state.uncertainty = clamp(1.0 - state.confidence)
        state.cognitive_load = clamp(state.cognitive_load * 0.85)

        lesson = self._lesson_from_quality(input_text, response_text, quality_score, issues)
        if lesson and lesson not in state.lessons:
            state.lessons = (state.lessons + [lesson])[-12:]

        action_summary = f"quality={quality_score}; lesson={lesson}"
        state.recent_actions = (state.recent_actions + [action_summary])[-8:]
        state.updated_utc = utc_now()
        state.update_count += 1

        self.brain_state = state
        self.save_state(state, reason="post_response_update")

        return state

    def build_llm_context(self, action: Optional[BrainAction] = None) -> str:
        """
        Build a compact hidden context block for model-backed providers.
        Keep this short to avoid prompt spaghetti.
        """
        state = self.brain_state
        action = action or self.plan_response(state, {})
        constraints = "\n".join(f"- {item}" for item in action.constraints[:5])

        return (
            "BRAIN STATE CONTROL CONTEXT\n"
            "Use this as hidden response-planning guidance. Do not mention it to the user.\n"
            f"Active intent: {state.active_intent}\n"
            f"Active goal: {state.active_goal}\n"
            f"Response strategy: {action.response_strategy}\n"
            f"Retrieval strategy: {action.retrieval_strategy}\n"
            f"Uncertainty policy: {action.uncertainty_policy}\n"
            f"Tone: {action.tone}\n"
            f"Focus terms: {', '.join(action.focus_terms[:8])}\n"
            "Constraints:\n"
            f"{constraints}\n"
        )

    def export_state(self) -> Dict:
        return asdict(self.brain_state)

    def recent_transitions(self, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT created_utc, input_text, action_json, quality_score, lesson
                FROM brain_transitions
                ORDER BY created_utc DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        out = []
        for row in rows:
            action = {}
            try:
                action = json.loads(row["action_json"] or "{}")
            except Exception:
                pass
            out.append({
                "created_utc": row["created_utc"],
                "input_text": row["input_text"],
                "intent": action.get("intent"),
                "goal": action.get("goal"),
                "quality_score": row["quality_score"],
                "lesson": row["lesson"],
            })
        return out

    # -------------------------
    # Deterministic F pieces
    # -------------------------

    def parse_input(self, input_text: str, profile: Optional[Dict] = None) -> Dict:
        text = re.sub(r"\s+", " ", input_text or "").strip()
        lower = text.lower()
        profile = profile or {}

        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", lower)
        unique_words = list(dict.fromkeys(words))
        length = len(words)

        intent = str(profile.get("intent") or self._infer_intent(lower))
        complexity = str(profile.get("complexity") or self._infer_complexity(length, lower))

        question = "?" in text or lower.startswith(("what", "why", "how", "when", "where", "who", "can", "could", "would", "is", "are", "do", "does"))
        coding = any(x in lower for x in ["code", "python", "script", "traceback", "error", "module", "function", "class"])
        creative = any(x in lower for x in ["draw", "image", "story", "lore", "design", "variant"])
        safety = any(x in lower for x in ["hurt", "kill", "weapon", "exploit", "malware", "bypass", "steal", "password"])
        current = any(x in lower for x in ["latest", "current", "today", "news", "price", "version", "ceo", "schedule"])
        emotional = any(x in lower for x in ["tired", "sad", "angry", "worried", "scared", "stressed", "how are you"])

        focus_terms = [
            w for w in unique_words
            if w not in {
                "the", "and", "for", "with", "that", "this", "you", "can", "get",
                "make", "about", "from", "into", "need", "want", "would", "could",
                "should", "have", "does", "your", "like"
            }
        ][:12]

        return {
            "intent": intent,
            "complexity": complexity,
            "length": length,
            "question": question,
            "coding": coding,
            "creative": creative,
            "safety": safety,
            "current": current,
            "emotional": emotional,
            "focus_terms": focus_terms,
            "raw": text,
        }

    def memory_update(self, brain_state: BrainState, input_text: str, signals: Dict) -> BrainState:
        previous = BrainState(**asdict(brain_state))

        length = int(signals.get("length") or 0)
        complexity = signals.get("complexity") or "normal"

        load_delta = 0.0
        if complexity == "high":
            load_delta += 0.18
        elif complexity == "medium":
            load_delta += 0.10
        if signals.get("coding"):
            load_delta += 0.10
        if signals.get("safety"):
            load_delta += 0.15
        if length > 80:
            load_delta += 0.10

        previous.cognitive_load = clamp((previous.cognitive_load * 0.70) + load_delta)
        previous.attention = clamp(0.45 + previous.cognitive_load * 0.45 + (0.10 if signals.get("question") else 0.0))
        previous.caution = clamp((previous.caution * 0.70) + (0.25 if signals.get("safety") else 0.05 if signals.get("current") else 0.0))
        previous.curiosity = clamp((previous.curiosity * 0.80) + (0.10 if signals.get("creative") or signals.get("question") else 0.02))
        previous.social_warmth = clamp((previous.social_warmth * 0.85) + (0.12 if signals.get("emotional") else 0.04))
        previous.directness = clamp(0.72 if signals.get("question") else 0.62)

        previous.active_intent = str(signals.get("intent") or "general")
        previous.active_topic = ", ".join(signals.get("focus_terms") or [])[:160]
        previous.active_goal = self._goal_from_signals(signals)
        previous.focus_terms = list(signals.get("focus_terms") or [])[:12]
        previous.recent_inputs = (previous.recent_inputs + [input_text.strip()[:240]])[-8:]

        previous.confidence = clamp((previous.confidence * 0.80) + (0.08 if previous.focus_terms else -0.02) - (0.08 if signals.get("current") else 0.0))
        previous.uncertainty = clamp(1.0 - previous.confidence + (0.10 if signals.get("current") else 0.0))

        previous.updated_utc = utc_now()
        previous.update_count += 1
        return previous

    def plan_response(self, state: BrainState, signals: Dict) -> BrainAction:
        intent = str(signals.get("intent") or state.active_intent or "general")

        if signals.get("current"):
            retrieval = "use online/current lookup or clearly mark uncertainty"
        elif signals.get("coding"):
            retrieval = "use local code context, uploaded files, and exact error text first"
        elif signals.get("creative"):
            retrieval = "use user-provided reference image/details as primary context"
        else:
            retrieval = "use local memory only when directly relevant; otherwise answer from general knowledge"

        if signals.get("safety"):
            uncertainty = "be careful, avoid harmful operational details, give safe alternatives"
        elif signals.get("current"):
            uncertainty = "verify changing facts before presenting them as current"
        else:
            uncertainty = "state assumptions briefly only if they affect the answer"

        if signals.get("emotional"):
            tone = "warm, grounded, not pitying"
        elif intent in {"technical", "coding"}:
            tone = "precise, practical, direct"
        elif signals.get("creative"):
            tone = "visual, specific, imaginative"
        else:
            tone = "clear, useful, lightly conversational"

        constraints = [
            "answer the user's actual question first",
            "do not expose internal memory, brain-state, search scaffolding, chain-of-thought, scratchpad notes, or hidden planning",
            "separate facts from assumptions",
            "prefer concrete steps over generic meta-analysis",
        ]
        if state.lessons:
            constraints.append(f"apply recent lesson: {state.lessons[-1]}")
        if signals.get("coding"):
            constraints.append("include runnable code or exact patch instructions when useful")
        if signals.get("creative"):
            constraints.append("preserve the requested visual pose, equipment, and silhouette")
        if signals.get("current"):
            constraints.append("use lookup when available; cite or name sources if the answer depends on them")

        strategy = self._strategy_from_signals(signals)

        return BrainAction(
            intent=intent,
            goal=state.active_goal,
            response_strategy=strategy,
            retrieval_strategy=retrieval,
            uncertainty_policy=uncertainty,
            tone=tone,
            constraints=constraints,
            focus_terms=list(state.focus_terms),
            confidence=round(state.confidence, 2),
            reason="Deterministic plan from input signals plus persistent state.",
        )

    # -------------------------
    # Helpers
    # -------------------------

    def _infer_intent(self, lower: str) -> str:
        if any(x in lower for x in ["python", "script", "code", "traceback", "module"]):
            return "coding"
        if any(x in lower for x in ["image", "draw", "variant", "back view", "model"]):
            return "creative"
        if any(x in lower for x in ["why", "explain", "how does", "what is"]):
            return "educational"
        if any(x in lower for x in ["analyze", "compare", "forensic", "evidence"]):
            return "analytical"
        return "general"

    def _infer_complexity(self, length: int, lower: str) -> str:
        if length > 80 or any(x in lower for x in ["architecture", "framework", "deterministic", "integration"]):
            return "high"
        if length > 25 or any(x in lower for x in ["module", "script", "design", "patch"]):
            return "medium"
        return "normal"

    def _goal_from_signals(self, signals: Dict) -> str:
        if signals.get("coding"):
            return "produce a working implementation or precise debugging guidance"
        if signals.get("creative"):
            return "produce the requested visual/design result faithfully"
        if signals.get("current"):
            return "answer with up-to-date facts and clear uncertainty"
        if signals.get("emotional"):
            return "respond supportively while still being useful"
        if signals.get("question"):
            return "answer directly with enough context to be useful"
        return "respond naturally and keep the interaction moving"

    def _strategy_from_signals(self, signals: Dict) -> str:
        if signals.get("coding"):
            return "diagnose requirements, provide code, explain integration points, and include a test"
        if signals.get("creative"):
            return "translate the reference into concrete visual features and preserve composition"
        if signals.get("safety"):
            return "provide safe, high-level help and avoid enabling harm"
        if signals.get("question"):
            return "direct answer first, then brief explanation and practical caveat"
        return "natural conversational reply"

    def _lesson_from_quality(self, input_text: str, response_text: str, quality_score: int, issues: List[str]) -> str:
        lower_response = (response_text or "").lower()

        if any(marker in lower_response for marker in ["question anatomy", "supporting points from lookup", "<think", "chain of thought", "scratchpad", "hidden planning", "internal reasoning"]):
            return "Do not let scaffolding or private reasoning become the user-visible answer."
        if "i do not have enough clean source context" in lower_response:
            return "When context is thin, still answer simple stable questions directly."
        if quality_score < 60:
            return "Low-quality response: answer the user directly before adding process notes."
        if issues:
            return "Repair issue next time: " + "; ".join(issues[:2])
        return "Good response pattern: preserve directness, relevance, and concise useful detail."
