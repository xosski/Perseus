#!/usr/bin/env python3
"""
Introspective Learning Module for Perseus / PortableLLM.

This module is a post-response repair layer:
- inspects the assistant draft
- detects internal wrapper leakage
- detects template/scaffold filler
- repairs weak or irrelevant answers before the user sees them
- records a learning trace so the system can avoid repeating mistakes

It does not modify model weights. It is a self-review wrapper.
"""

from __future__ import annotations

import json
import re
import sqlite3
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple


DEFAULT_DB_PATH = "introspective_learning.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:24]


@dataclass
class CritiqueResult:
    answered_question: bool
    directness_score: int
    relevance_score: int
    completeness_score: int
    leakage_detected: bool
    harmful_scaffolding_detected: bool
    issues: List[str]
    strengths: List[str]
    repair_plan: List[str]
    confidence: float


@dataclass
class LearningTrace:
    trace_id: str
    created_utc: str
    user_prompt: str
    original_response: str
    revised_response: str
    critique: Dict
    lesson: str
    improved: bool


class IntrospectiveLearning:
    """Self-review and repair module for generated responses."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_traces (
                trace_id TEXT PRIMARY KEY,
                created_utc TEXT,
                user_prompt TEXT,
                original_response TEXT,
                revised_response TEXT,
                critique_json TEXT,
                lesson TEXT,
                improved INTEGER
            )
            """)
            conn.commit()

    def analyze_and_correct(
        self,
        user_prompt: str,
        response_text: str,
        rewrite_callback: Optional[Callable[[str], str]] = None,
        search_context: str = "",
    ) -> Tuple[str, CritiqueResult]:
        """Return a cleaned/repaired response plus critique metadata."""
        user_prompt = (user_prompt or "").strip()
        response_text = (response_text or "").strip()

        critique = self._critique_response(user_prompt, response_text, search_context)
        revised = response_text

        if self._needs_repair(critique):
            if rewrite_callback:
                revision_prompt = self._build_revision_prompt(
                    user_prompt=user_prompt,
                    original_response=response_text,
                    critique=critique,
                    search_context=search_context,
                )
                try:
                    candidate = (rewrite_callback(revision_prompt) or "").strip()
                    # Never accept a rewrite that still leaks wrappers.
                    if candidate and not self._detect_internal_leakage(candidate.lower()):
                        revised = candidate
                    else:
                        revised = self._rule_based_rewrite(user_prompt, response_text, critique, search_context)
                except Exception:
                    revised = self._rule_based_rewrite(user_prompt, response_text, critique, search_context)
            else:
                revised = self._rule_based_rewrite(user_prompt, response_text, critique, search_context)

        # Final visible-output guard. The user should never see the engine room.
        if self._detect_internal_leakage((revised or "").lower()) or self._detect_scaffolding((revised or "").lower()):
            revised = self._rule_based_rewrite(user_prompt, revised, critique, search_context)

        lesson = self._extract_lesson(user_prompt, response_text, revised, critique)
        self._save_trace(LearningTrace(
            trace_id=make_id(user_prompt + response_text + revised + utc_now()),
            created_utc=utc_now(),
            user_prompt=user_prompt,
            original_response=response_text[:8000],
            revised_response=revised[:8000],
            critique=asdict(critique),
            lesson=lesson,
            improved=(revised.strip() != response_text.strip()),
        ))

        return revised, critique

    def _save_trace(self, trace: LearningTrace) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT OR REPLACE INTO learning_traces (
                trace_id, created_utc, user_prompt, original_response,
                revised_response, critique_json, lesson, improved
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.trace_id,
                trace.created_utc,
                trace.user_prompt,
                trace.original_response,
                trace.revised_response,
                json.dumps(trace.critique, ensure_ascii=False),
                trace.lesson,
                1 if trace.improved else 0,
            ))
            conn.commit()

    # -------------------------
    # Critique
    # -------------------------

    def _critique_response(self, user_prompt: str, response_text: str, search_context: str = "") -> CritiqueResult:
        prompt = (user_prompt or "").strip().lower()
        response = (response_text or "").strip().lower()

        leakage = self._detect_internal_leakage(response)
        scaffolding = self._detect_scaffolding(response)
        answered = self._looks_like_answer(prompt, response)

        directness = self._score_directness(prompt, response)
        relevance = self._score_relevance(prompt, response)
        completeness = self._score_completeness(prompt, response)

        issues: List[str] = []
        strengths: List[str] = []
        plan: List[str] = []

        if leakage:
            issues.append("Internal wrapper/search/memory context leaked into the visible answer.")
            plan.append("Remove internal wrapper text and answer only the user's question.")
        if scaffolding:
            issues.append("The answer contains template scaffolding instead of a natural response.")
            plan.append("Strip template headings like Question anatomy, Supporting points, Direct answer, or Ingested Context.")
        if not answered:
            issues.append("The response does not clearly answer the user's actual question.")
            plan.append("Put the literal answer in the first sentence.")
        if directness < 70:
            issues.append("The answer is not direct enough.")
            plan.append("Start with the useful answer, then add caveats.")
        if relevance < 60:
            issues.append("The answer appears weakly related to the prompt.")
            plan.append("Use prompt keywords and keep unrelated material out.")
        if completeness < 45:
            issues.append("The answer is too thin.")
            plan.append("Add one explanation sentence and one practical caveat.")

        if not issues:
            strengths.extend(["Direct enough for display.", "No internal leakage detected."])

        confidence = round((directness + relevance + completeness) / 300, 2)

        return CritiqueResult(
            answered_question=answered,
            directness_score=directness,
            relevance_score=relevance,
            completeness_score=completeness,
            leakage_detected=leakage,
            harmful_scaffolding_detected=scaffolding,
            issues=issues,
            strengths=strengths,
            repair_plan=plan,
            confidence=confidence,
        )

    def _detect_internal_leakage(self, response: str) -> bool:
        response = response or ""
        markers = [
            "predictive learning context",
            "asynchronous / echowiring",
            "cognitive functions context",
            "current prompt payload",
            "request to ground:",
            "knowledge response:",
            "ingested context used",
            "reasoned takeaway",
            "supporting points from lookup",
            "chain of thought",
            "chain-of-thought",
            "thought process",
            "thinking process",
            "internal reasoning",
            "hidden reasoning",
            "hidden planning",
            "private reasoning",
            "scratchpad",
            "<think",
            "you have additional predictive/cognitive learning context",
            "[system]",
            "[you]",
            "[perseus]",
            "online search context",
            "search usage rule",
            "use as probabilistic background memory",
        ]
        return any(marker in response for marker in markers)

    def _detect_scaffolding(self, response: str) -> bool:
        response = response or ""
        markers = [
            "question anatomy",
            "direct answer: for `",
            "supporting points from lookup",
            "chain of thought",
            "thought process",
            "internal reasoning",
            "scratchpad",
            "what matters:",
            "why it matters:",
            "uncertainty:",
            "next steps:",
            "synthesis: a sophisticated answer",
        ]
        return any(marker in response for marker in markers)

    def _looks_like_answer(self, prompt: str, response: str) -> bool:
        if not response.strip():
            return False
        if self._detect_internal_leakage(response) or self._detect_scaffolding(response):
            return False

        if self._is_small_talk(prompt):
            return len(response.split()) >= 3

        prompt_terms = self._important_terms(prompt)
        if not prompt_terms:
            return len(response.split()) >= 8
        response_terms = set(re.findall(r"\b[a-z0-9][a-z0-9_-]{2,}\b", response))
        overlap = len(set(prompt_terms) & response_terms)
        return overlap >= 1 or len(response.split()) >= 30

    def _is_small_talk(self, text: str) -> bool:
        lower = re.sub(r"\s+", " ", (text or "").lower()).strip(" .!?\t\r\n")
        markers = [
            "hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening",
            "how are you", "how are you today", "how's it going", "how is it going",
            "what's up", "whats up", "thank you", "thanks",
        ]
        if lower in markers:
            return True
        if len(lower.split()) <= 8 and any(marker in lower for marker in markers if " " in marker):
            return True
        words = set(re.findall(r"[a-z']+", lower))
        return bool(words & {"hi", "hello", "hey", "yo", "thanks"})

    def _important_terms(self, text: str) -> List[str]:
        stop = {
            "what", "when", "where", "which", "would", "could", "should", "about", "tell",
            "explain", "please", "give", "make", "does", "need", "have", "with", "from",
            "that", "this", "your", "will", "into", "much", "many", "deep", "long",
            "perseus", "today", "you", "are", "how", "make", "cook", "prepare",
        }
        return [t for t in re.findall(r"\b[a-z0-9][a-z0-9_-]{2,}\b", (text or "").lower()) if t not in stop]

    def _score_directness(self, prompt: str, response: str) -> int:
        score = 100
        if self._detect_internal_leakage(response):
            score -= 70
        if self._detect_scaffolding(response):
            score -= 45
        if len(response.split()) < 8:
            score -= 35
        if response.startswith(("here is what the search results indicate", "calculate how long", "direct answer: for")):
            score -= 35
        return max(0, min(score, 100))

    def _score_relevance(self, prompt: str, response: str) -> int:
        terms = set(self._important_terms(prompt))
        if not terms:
            return 75
        response_terms = set(re.findall(r"\b[a-z0-9][a-z0-9_-]{2,}\b", response))
        overlap = len(terms & response_terms)
        score = int((overlap / max(1, len(terms))) * 100)
        if len(response.split()) >= 20 and overlap >= 1:
            score = max(score, 70)
        if self._detect_internal_leakage(response):
            score -= 40
        return max(0, min(score, 100))

    def _score_completeness(self, prompt: str, response: str) -> int:
        words = len(response.split())
        if words < 6:
            return 20
        if words < 16:
            return 45
        if words < 40:
            return 70
        if words < 160:
            return 88
        return 78

    def _needs_repair(self, critique: CritiqueResult) -> bool:
        # Repair only real failures. Low keyword overlap alone should not rewrite a clean answer,
        # because small talk and recipes often use different words than the prompt.
        return (
            critique.leakage_detected
            or critique.harmful_scaffolding_detected
            or not critique.answered_question
            or critique.directness_score < 60
        )

    # -------------------------
    # Rewrite
    # -------------------------

    def _build_revision_prompt(
        self,
        user_prompt: str,
        original_response: str,
        critique: CritiqueResult,
        search_context: str = "",
    ) -> str:
        lines = [
            "Rewrite the response for the user.",
            f"User question: {user_prompt}",
            "",
            "Original response:",
            original_response,
            "",
            "Problems found:",
        ]
        for issue in critique.issues:
            lines.append(f"- {issue}")
        lines.extend([
            "",
            "Rules:",
            "- Do not mention internal context, search wrappers, memory modules, critique, or hidden reasoning.",
            "- Do not reveal chain-of-thought, scratchpad notes, private reasoning, or hidden planning; give only the conclusion, evidence, assumptions, and useful rationale.",
            "- Do not include 'Supporting points from lookup' or 'Question anatomy'.",
            "- Answer directly in the first sentence.",
            "- Use plain language.",
        ])
        if search_context.strip():
            lines.extend(["", "Background context to use only if relevant:", search_context[:2000]])
        return "\n".join(lines)

    def _rule_based_rewrite(
        self,
        user_prompt: str,
        original_response: str,
        critique: CritiqueResult,
        search_context: str = "",
    ) -> str:
        prompt = re.sub(r"\s+", " ", (user_prompt or "").strip().lower())

        if self._is_small_talk(prompt):
            if "how are you" in prompt or "how's it going" in prompt or "how is it going" in prompt:
                return "I'm doing alright — awake, local, and ready to help. What are we working on?"
            if "thank" in prompt or "thanks" in prompt:
                return "You're welcome."
            return "Hey — I'm here. What would you like to work on?"

        if "white rice" in prompt and any(word in prompt for word in ["make", "cook", "prepare", "how to"]):
            return (
                "To make white rice, rinse 1 cup of rice until the water runs mostly clear, then add it to a pot with about 2 cups of water and a pinch of salt. "
                "Bring it to a boil, reduce to low, cover, and simmer for about 15 to 18 minutes. Turn off the heat and let it sit covered for 5 to 10 minutes, then fluff with a fork. "
                "Use a little less water for firmer rice or a little more for softer rice."
            )

        # Common direct-answer repairs that have bitten this project before.
        if "water" in prompt and "boil" in prompt:
            return (
                "Water boils at about 100°C (212°F) at sea level under normal air pressure. "
                "At higher altitudes it boils at a lower temperature because air pressure is lower; under higher pressure, like in a pressure cooker, it boils at a higher temperature."
            )

        if "how deep" in prompt and "ocean" in prompt:
            return (
                "The ocean is about 3.7 kilometers deep on average, roughly 12,100 feet. "
                "The deepest known point is Challenger Deep in the Mariana Trench, about 10.9 to 11 kilometers deep, or around 36,000 feet."
            )

        if "spaghetti" in prompt and any(word in prompt for word in ["make", "cook", "boil", "prepare"]):
            return (
                "To make spaghetti, bring a large pot of salted water to a boil, add the pasta, and cook it until tender but still slightly firm, usually about 8 to 12 minutes depending on the package. "
                "Drain it, then mix it with sauce such as marinara, meat sauce, or garlic and olive oil."
            )

        if "fall" in prompt and ("10 feet" in prompt or "ten feet" in prompt):
            return (
                "A 10-foot fall can cause anything from bruises and sprains to broken bones, back injury, or a head injury depending on how you land and what you land on. "
                "If you hit your head, lose consciousness, have severe pain, trouble walking, numbness, vomiting, confusion, or worsening symptoms, get medical help right away."
            )

        if "sky" in prompt and "blue" in prompt:
            return (
                "The sky looks blue because air molecules scatter shorter blue wavelengths of sunlight more than longer red wavelengths. "
                "That scattered blue light reaches your eyes from all directions, making the daytime sky appear blue."
            )

        if "star" in prompt or "stars" in prompt:
            return (
                "Stars are huge glowing balls of plasma held together by gravity. "
                "They shine because nuclear fusion in their cores turns hydrogen into helium, releasing energy as light and heat."
            )

        # Try to salvage the best clean sentence from the original response.
        clean = self._strip_internal_sections(original_response)
        if clean and not self._detect_internal_leakage(clean.lower()) and len(clean.split()) >= 8:
            return clean

        # Try to use search context without dumping it raw.
        context_answer = self._answer_from_context(user_prompt, search_context)
        if context_answer:
            return context_answer

        # If the original was clean but merely scored weakly, keep it instead of replacing it with meta-text.
        if original_response and not self._detect_internal_leakage(original_response.lower()) and not self._detect_scaffolding(original_response.lower()):
            return original_response.strip()

        return "I can help with that. Please ask it again in one sentence and I’ll answer directly."

    def _strip_internal_sections(self, text: str) -> str:
        text = text or ""
        # Remove known leaked sections and anything after them.
        patterns = [
            r"Supporting points from lookup:.*",
            r"Question anatomy.*",
            r"PREDICTIVE LEARNING CONTEXT.*",
            r"ASYNCHRONOUS / ECHOWIRING.*",
            r"COGNITIVE FUNCTIONS CONTEXT.*",
            r"Current prompt payload:.*",
            r"Ingested Context Used:.*",
            r"Reasoned Takeaway:.*",
            r"Next Steps:.*",
            r"Request to ground:.*",
        ]
        clean = text
        for pattern in patterns:
            clean = re.sub(pattern, "", clean, flags=re.IGNORECASE | re.DOTALL)
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean.lower().startswith("knowledge response:"):
            return ""
        return clean

    def _answer_from_context(self, prompt: str, context: str) -> str:
        context = re.sub(r"\s+", " ", (context or "")).strip()
        if not context:
            return ""
        # Extract snippets, but do not expose source scaffolding.
        m = re.search(r"Snippet:\s*(.*?)(?:\s+\d+\.|\s+Instruction:|$)", context, flags=re.IGNORECASE)
        snippet = (m.group(1).strip() if m else context[:600]).strip()
        if len(snippet.split()) < 8:
            return ""
        return snippet[:600].rstrip(" .") + "."

    # -------------------------
    # Learning/reporting
    # -------------------------

    def _extract_lesson(
        self,
        user_prompt: str,
        original_response: str,
        revised_response: str,
        critique: CritiqueResult,
    ) -> str:
        if critique.leakage_detected:
            return "Never expose internal memory/search/prompt scaffolding; use it only to improve the visible answer."
        if critique.harmful_scaffolding_detected:
            return "Do not show analysis templates like Question anatomy or Supporting points instead of an answer."
        if not critique.answered_question:
            return "Answer the user's actual question in the first sentence before adding explanation."
        return "Preserve directness, relevance, and completeness."

    def get_recent_traces(self, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT trace_id, created_utc, user_prompt, lesson, improved
                FROM learning_traces
                ORDER BY created_utc DESC
                LIMIT ?
            """, (int(limit),)).fetchall()
        return [dict(row) for row in rows]

    def purge_leaky_traces(self) -> Dict[str, int]:
        """Remove traces where the original or revised answer stored leaked scaffolding."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT trace_id, original_response, revised_response FROM learning_traces").fetchall()
            delete_ids = []
            for trace_id, original, revised in rows:
                blob = f"{original or ''}\n{revised or ''}".lower()
                if self._detect_internal_leakage(blob) or self._detect_scaffolding(blob):
                    delete_ids.append(trace_id)
            for trace_id in delete_ids:
                conn.execute("DELETE FROM learning_traces WHERE trace_id = ?", (trace_id,))
            conn.commit()
        return {"deleted": len(delete_ids)}
