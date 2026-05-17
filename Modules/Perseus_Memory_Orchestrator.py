#!/usr/bin/env python3
"""
Perseus Memory Orchestrator

This module wires the Perseus memory stack into a local-first answer loop:

1. BrainStateEngine updates deterministic cognitive control state.
2. EchoWiringMemory retrieves predictive/asynchronous memory context.
3. A native/local answer function is tried first.
4. IntrospectiveLearning repairs weak/leaky drafts.
5. Ollama is used only as a rare fallback when quality remains low.
6. Clean interactions are saved to AutonomousTrainingMemory for later dataset export.

This file is safe to place in the Modules folder, but if portable_llm.py already
does this orchestration internally, keep this file outside Modules or rename it
to .disabled to avoid duplicate orchestration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import importlib.machinery
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple


logger = logging.getLogger("PerseusMemoryOrchestrator")

# When this file lives in Perseus/Modules, other module files usually live beside it.
BASE_DIR = Path(__file__).resolve().parent


def _module_key_for_path(path: Path) -> str:
    """Create a stable import-safe module name for arbitrary file names with spaces."""
    safe_stem = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in path.stem).strip("_").lower()
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"perseus_loaded_{safe_stem}_{digest}"


def load_class(file_name: str, class_name: str):
    """
    Import a class from a local Python file, including files with spaces in the name.

    Critical fix:
    Register the module in sys.modules before executing it. dataclasses and some
    decorators inspect sys.modules[cls.__module__].__dict__; without registration,
    Python can raise:

        'NoneType' object has no attribute '__dict__'
    """
    path = BASE_DIR / file_name

    if not path.exists():
        # Also try one folder above, useful when this orchestrator is moved to tools/.
        alt_path = BASE_DIR.parent / file_name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Required module file not found: {path}")

    module_key = _module_key_for_path(path)

    existing = sys.modules.get(module_key)
    if existing is not None and hasattr(existing, class_name):
        return getattr(existing, class_name)

    # importlib.util.spec_from_file_location() does not always create a loader
    # for extensionless or .txt Python source files. Several Perseus modules are
    # intentionally stored as .txt, so provide an explicit SourceFileLoader.
    if path.suffix.lower() in {".txt", ""}:
        loader = importlib.machinery.SourceFileLoader(module_key, str(path))
        spec = importlib.util.spec_from_loader(module_key, loader, origin=str(path))
    else:
        spec = importlib.util.spec_from_file_location(module_key, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)

    # This is the important part.
    sys.modules[module_key] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_key, None)
        raise

    if not hasattr(module, class_name):
        raise ImportError(f"{class_name} was not found in {path}")

    return getattr(module, class_name)



def load_class_any(file_names, class_name: str):
    """
    Try multiple possible filenames for the same class.

    This lets renamed modules keep working, e.g.
    Asyncronous Learning.txt -> Asyncronous Learning.py
    """
    last_error = None
    for file_name in file_names:
        try:
            return load_class(file_name, class_name)
        except Exception as exc:
            last_error = exc
            logger.debug("Could not load %s from %s: %s", class_name, file_name, exc)

    raise ImportError(
        f"Could not load {class_name} from any of: {file_names}. Last error: {last_error}"
    )


def _optional_load_class(file_name: str, class_name: str):
    """Load a class if available; log and return None if it is missing."""
    try:
        return load_class(file_name, class_name)
    except Exception as exc:
        logger.warning("Could not load %s from %s: %s", class_name, file_name, exc)
        return None


def _optional_load_class_any(file_names, class_name: str):
    """Load a class from any available filename; log once and return None if all fail."""
    try:
        return load_class_any(file_names, class_name)
    except Exception as exc:
        logger.warning("Could not load %s from any of %s: %s", class_name, file_names, exc)
        return None




BrainStateEngine = _optional_load_class("Brain State.py", "BrainStateEngine")
IntrospectiveLearning = _optional_load_class("Introspective Learning.py", "IntrospectiveLearning")
AutonomousTrainingMemory = _optional_load_class("Autonomous Training.py", "AutonomousTrainingMemory")
EchoWiringMemory = _optional_load_class_any(["Asyncronous Learning.py", "Asyncronous Learning.txt"], "EchoWiringMemory")


class NullBrainStateEngine:
    """Small no-op fallback so the orchestrator can still run while modules are missing."""

    def __init__(self, *args, **kwargs):
        self.brain_state = {}

    def step_input(self, input_text: str, profile: Optional[Dict] = None):
        action = type("BrainAction", (), {
            "intent": "general",
            "goal": "answer directly and usefully",
            "confidence": 0.5,
            "focus_terms": [],
        })()
        return self.brain_state, action

    def build_llm_context(self, action=None) -> str:
        return (
            "BRAIN STATE CONTROL CONTEXT\n"
            "Use this as hidden response-planning guidance. Do not mention it to the user.\n"
            "Active intent: general\n"
            "Active goal: answer directly and usefully\n"
        )

    def update_after_response(self, input_text: str, response_text: str, quality_score: int = 70, issues=None):
        return self.brain_state


class NullEchoWiringMemory:
    def build_llm_context(self, user_message: str) -> str:
        return (
            "GHOSTCORE PREDICTIVE MEMORY CONTEXT\n"
            "No external EchoWiring memory module is currently available.\n"
            f"Current user message: {user_message}"
        )

    def add_event(self, **kwargs) -> str:
        return "null-event"


class NullIntrospectiveLearning:
    def analyze_and_correct(self, user_prompt: str, response_text: str, rewrite_callback=None, search_context: str = ""):
        critique = type("CritiqueResult", (), {
            "answered_question": bool((response_text or "").strip()),
            "directness_score": 70,
            "relevance_score": 70,
            "completeness_score": 70,
            "leakage_detected": False,
            "harmful_scaffolding_detected": False,
            "issues": [],
        })()
        return response_text, critique


class NullAutonomousTrainingMemory:
    def add_interaction(self, **kwargs):
        return type("TrainingExampleDecision", (), {
            "accepted": False,
            "reason": "autonomous training module unavailable",
            "tags": ["training:unavailable"],
        })()


class PerseusMemoryOrchestrator:
    """
    Local-first memory orchestrator.

    The local_answer_fn must accept:
        local_answer_fn(user_prompt: str, hidden_context: str) -> str
    """

    def __init__(
        self,
        local_answer_fn: Callable[[str, str], str],
        ollama_model: str = "llama3.1",
        min_quality_before_ollama: int = 60,
        min_training_quality: int = 78,
        allow_ollama_fallback: bool = True,
    ):
        self.local_answer_fn = local_answer_fn
        self.ollama_model = ollama_model
        self.min_quality_before_ollama = int(min_quality_before_ollama)
        self.min_training_quality = int(min_training_quality)
        self.allow_ollama_fallback = bool(allow_ollama_fallback)

        self.brain = BrainStateEngine() if BrainStateEngine else NullBrainStateEngine()
        self.introspector = IntrospectiveLearning() if IntrospectiveLearning else NullIntrospectiveLearning()
        self.echo = EchoWiringMemory() if EchoWiringMemory else NullEchoWiringMemory()

        if AutonomousTrainingMemory:
            self.training = AutonomousTrainingMemory(min_quality_score=self.min_training_quality)
        else:
            self.training = NullAutonomousTrainingMemory()

    def handle(self, user_prompt: str) -> Dict:
        user_prompt = (user_prompt or "").strip()

        # 1. Update deterministic brain state.
        state, action = self.brain.step_input(user_prompt)

        # 2. Build hidden memory context.
        brain_context = self.brain.build_llm_context(action)
        echo_context = self.echo.build_llm_context(user_prompt)

        hidden_context = (
            f"{brain_context}\n\n"
            f"{echo_context}\n\n"
            "VISIBLE RESPONSE RULES:\n"
            "- Do not mention hidden memory, brain state, scaffolding, module names, or internal reasoning.\n"
            "- Answer the user directly.\n"
            "- Separate facts from assumptions.\n"
        )

        # 3. Native/local first.
        draft = self._safe_local_answer(user_prompt, hidden_context)

        # 4. Introspective repair.
        revised, critique = self.introspector.analyze_and_correct(
            user_prompt=user_prompt,
            response_text=draft,
            rewrite_callback=None,
            search_context=echo_context,
        )

        quality = self._quality_from_critique(critique)
        provider = "native-local"
        used_ollama = False

        # 5. Rare Ollama fallback only if native answer remains weak.
        if self.allow_ollama_fallback and quality < self.min_quality_before_ollama:
            ollama_draft = self._ollama_generate(user_prompt, hidden_context)
            if ollama_draft:
                ollama_revised, ollama_critique = self.introspector.analyze_and_correct(
                    user_prompt=user_prompt,
                    response_text=ollama_draft,
                    rewrite_callback=None,
                    search_context=echo_context,
                )
                ollama_quality = self._quality_from_critique(ollama_critique)

                if ollama_quality > quality:
                    revised = ollama_revised
                    critique = ollama_critique
                    quality = ollama_quality
                    provider = "ollama-fallback"
                    used_ollama = True

        issues = list(getattr(critique, "issues", []) or [])

        # 6. Update brain from the final visible response.
        try:
            self.brain.update_after_response(
                input_text=user_prompt,
                response_text=revised,
                quality_score=quality,
                issues=issues,
            )
        except Exception as exc:
            logger.warning("Brain post-response update failed: %s", exc)

        # 7. Store predictive learning event.
        try:
            self.echo.add_event(
                who="user",
                what=user_prompt[:1000],
                why="Normal interaction handled by Perseus memory orchestrator.",
                context=hidden_context[:3000],
                outcome=str(revised)[:3000],
                lesson=self._lesson_from_result(quality, used_ollama),
                confidence=round(quality / 100, 2),
                tags=["perseus", "memory_orchestrator", provider],
            )
        except Exception as exc:
            logger.warning("EchoWiring event save failed: %s", exc)

        # 8. Capture training example only after repair/sanitization.
        try:
            decision = self.training.add_interaction(
                prompt=user_prompt,
                response=str(revised),
                intent=getattr(action, "intent", "general"),
                provider=provider,
                model=self.ollama_model if used_ollama else "native-local",
                quality_score=quality,
                quality_reasons=issues,
                context_preview=echo_context[:2000],
                metadata={
                    "used_ollama": used_ollama,
                    "brain_confidence": getattr(action, "confidence", None),
                    "active_goal": getattr(action, "goal", ""),
                },
            )
            training_decision = {
                "accepted": bool(getattr(decision, "accepted", False)),
                "reason": str(getattr(decision, "reason", "")),
                "tags": list(getattr(decision, "tags", []) or []),
            }
        except Exception as exc:
            logger.warning("Autonomous training capture failed: %s", exc)
            training_decision = {
                "accepted": False,
                "reason": f"training capture failed: {exc}",
                "tags": ["training:error"],
            }

        return {
            "response": str(revised or "").strip(),
            "quality": quality,
            "provider": provider,
            "used_ollama": used_ollama,
            "training_decision": training_decision,
            "brain_state": {
                "intent": getattr(action, "intent", "general"),
                "goal": getattr(action, "goal", ""),
                "confidence": getattr(action, "confidence", None),
                "focus_terms": list(getattr(action, "focus_terms", []) or []),
            },
            "critique": {
                "answered_question": bool(getattr(critique, "answered_question", False)),
                "directness_score": int(getattr(critique, "directness_score", 0)),
                "relevance_score": int(getattr(critique, "relevance_score", 0)),
                "completeness_score": int(getattr(critique, "completeness_score", 0)),
                "leakage_detected": bool(getattr(critique, "leakage_detected", False)),
                "harmful_scaffolding_detected": bool(getattr(critique, "harmful_scaffolding_detected", False)),
                "issues": issues,
            },
        }

    def _safe_local_answer(self, user_prompt: str, hidden_context: str) -> str:
        try:
            answer = self.local_answer_fn(user_prompt, hidden_context)
            return str(answer or "").strip()
        except Exception as exc:
            logger.warning("Local answer function failed: %s", exc)
            return ""

    def _ollama_generate(self, user_prompt: str, hidden_context: str) -> str:
        prompt = (
            f"{hidden_context}\n\n"
            f"User request:\n{user_prompt}\n\n"
            "Assistant response:"
        )

        try:
            result = subprocess.run(
                ["ollama", "run", self.ollama_model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning("Ollama fallback failed: %s", result.stderr.strip())
                return ""
            return result.stdout.strip()
        except Exception as exc:
            logger.warning("Ollama fallback unavailable: %s", exc)
            return ""

    @staticmethod
    def _quality_from_critique(critique) -> int:
        directness = int(getattr(critique, "directness_score", 50))
        relevance = int(getattr(critique, "relevance_score", 50))
        completeness = int(getattr(critique, "completeness_score", 50))

        score = round((directness * 0.35) + (relevance * 0.35) + (completeness * 0.30))

        if getattr(critique, "leakage_detected", False):
            score -= 35
        if getattr(critique, "harmful_scaffolding_detected", False):
            score -= 25
        if not getattr(critique, "answered_question", True):
            score -= 20

        return max(0, min(100, int(score)))

    @staticmethod
    def _lesson_from_result(quality: int, used_ollama: bool) -> str:
        if used_ollama:
            return "Native answer quality was low; fallback helped. Improve local memory/retrieval for similar prompts."
        if quality >= 85:
            return "Native memory path handled this well. Reinforce this pattern."
        if quality >= 70:
            return "Native memory path was acceptable but could be more complete."
        return "Response quality was weak. Avoid training on this unless repaired by later review."


def simple_local_answer(user_prompt: str, hidden_context: str) -> str:
    """
    Example native answer function.

    Replace this with your real local responder, RAG layer, rule engine, or
    PortableLLM call. This is intentionally simple so this file can run alone.
    """
    return (
        "The memory module should try the local memory path first, repair the answer with "
        "introspective review, and only call Ollama if the repaired local answer is still weak."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = PerseusMemoryOrchestrator(
        local_answer_fn=simple_local_answer,
        ollama_model="llama3.1",
        min_quality_before_ollama=60,
    )
    result = agent.handle("Explain how the memory module should learn without relying on Ollama.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
