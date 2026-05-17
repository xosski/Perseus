#!/usr/bin/env python3
"""
Perseus Memory Orchestrator

Flow:
1. Update brain state from user input.
2. Retrieve predictive/asynchronous memory.
3. Generate a local-first answer.
4. Self-critique and repair.
5. Use Ollama only when quality is too low.
6. Save lessons and accepted training examples.
"""

from __future__ import annotations

import json
import subprocess
import importlib.util
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent


def load_class(file_name: str, class_name: str):
    """
    Allows importing your current files even though they have spaces in names.
    Later, rename them to normal module names for sanity:
    Brain State.py -> brain_state.py
    Autonomous Training.py -> autonomous_training.py
    """
    path = BASE_DIR / file_name
    spec = importlib.util.spec_from_file_location(class_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


BrainStateEngine = load_class("Brain State.py", "BrainStateEngine")
IntrospectiveLearning = load_class("Introspective Learning.py", "IntrospectiveLearning")
AutonomousTrainingMemory = load_class("Autonomous Training.py", "AutonomousTrainingMemory")
EchoWiringMemory = load_class("Asyncronous Learning.txt", "EchoWiringMemory")


class PerseusMemoryOrchestrator:
    def __init__(
        self,
        local_answer_fn: Callable[[str, str], str],
        ollama_model: str = "llama3.1",
        min_quality_before_ollama: int = 68,
        min_training_quality: int = 78,
    ):
        self.brain = BrainStateEngine()
        self.introspector = IntrospectiveLearning()
        self.training = AutonomousTrainingMemory(min_quality_score=min_training_quality)
        self.echo = EchoWiringMemory()

        self.local_answer_fn = local_answer_fn
        self.ollama_model = ollama_model
        self.min_quality_before_ollama = min_quality_before_ollama
        self.min_training_quality = min_training_quality

    def handle(self, user_prompt: str) -> Dict:
        # 1. Update active state.
        state, action = self.brain.step_input(user_prompt)

        # 2. Build memory context.
        brain_context = self.brain.build_llm_context(action)
        echo_context = self.echo.build_llm_context(user_prompt)

        hidden_context = (
            f"{brain_context}\n\n"
            f"{echo_context}\n\n"
            "VISIBLE RESPONSE RULES:\n"
            "- Do not mention hidden memory, brain state, scaffolding, or internal reasoning.\n"
            "- Answer the user directly.\n"
        )

        # 3. Try local/native answer first.
        draft = self.local_answer_fn(user_prompt, hidden_context)

        # 4. Critique and repair.
        revised, critique = self.introspector.analyze_and_correct(
            user_prompt=user_prompt,
            response_text=draft,
            rewrite_callback=None,
            search_context=echo_context,
        )

        quality = self._quality_from_critique(critique)

        provider = "native-local"
        used_ollama = False

        # 5. Rare Ollama fallback.
        if quality < self.min_quality_before_ollama:
            ollama_draft = self._ollama_generate(user_prompt, hidden_context)
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

        # 6. Update brain after final response.
        self.brain.update_after_response(
            input_text=user_prompt,
            response_text=revised,
            quality_score=quality,
            issues=getattr(critique, "issues", []),
        )

        # 7. Store predictive learning event.
        self.echo.add_event(
            who="user",
            what=user_prompt[:1000],
            why="Normal interaction handled by Perseus memory orchestrator.",
            context=hidden_context[:3000],
            outcome=revised[:3000],
            lesson=self._lesson_from_result(quality, used_ollama),
            confidence=round(quality / 100, 2),
            tags=["perseus", "memory_orchestrator", provider],
        )

        # 8. Capture training example only if clean enough.
        decision = self.training.add_interaction(
            prompt=user_prompt,
            response=revised,
            intent=action.intent,
            provider=provider,
            model=self.ollama_model if used_ollama else "native-local",
            quality_score=quality,
            quality_reasons=getattr(critique, "issues", []),
            context_preview=echo_context[:2000],
            metadata={
                "used_ollama": used_ollama,
                "brain_confidence": action.confidence,
                "active_goal": action.goal,
            },
        )

        return {
            "response": revised,
            "quality": quality,
            "provider": provider,
            "used_ollama": used_ollama,
            "training_decision": {
                "accepted": decision.accepted,
                "reason": decision.reason,
                "tags": decision.tags,
            },
            "brain_state": {
                "intent": action.intent,
                "goal": action.goal,
                "confidence": action.confidence,
                "focus_terms": action.focus_terms,
            },
        }

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
                return ""
            return result.stdout.strip()
        except Exception as exc:
            return f"I could not complete the fallback generation because Ollama failed: {exc}"

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

        return max(0, min(100, score))

    @staticmethod
    def _lesson_from_result(quality: int, used_ollama: bool) -> str:
        if used_ollama:
            return "Native answer quality was low; fallback helped. Improve local memory/retrieval for similar prompts."
        if quality >= 85:
            return "Native memory path handled this well. Reinforce this pattern."
        if quality >= 70:
            return "Native memory path was acceptable but could be more complete."
        return "Response quality was weak. Avoid training on this unless repaired by later review."


# Example local answer function.
# Replace this with your actual local model, rules engine, RAG engine, or native responder.
def simple_local_answer(user_prompt: str, hidden_context: str) -> str:
    return (
        "I can route this through the local memory system first, then only call Ollama "
        "if the self-review layer decides the answer is weak or incomplete."
    )


if __name__ == "__main__":
    agent = PerseusMemoryOrchestrator(
        local_answer_fn=simple_local_answer,
        ollama_model="llama3.1",
    )

    result = agent.handle("Explain how the memory module should learn without relying on Ollama.")
    print(json.dumps(result, indent=2))