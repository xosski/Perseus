#!/usr/bin/env python3
"""
Autonomous Training Module for Perseus / PortableLLM.

This module is the bridge from "Perseus learns around Ollama" to
"Perseus can eventually train and promote its own native model."

It does not fine-tune model weights directly by default. Instead it:
- captures high-quality local interactions as supervised training examples
- rejects low-quality, leaky, or fallback-only responses
- exports clean JSONL datasets for LoRA/SFT fine-tuning tools
- records training run metadata and candidate model promotion decisions

The actual deep-learning training step should be run by an explicit trainer
such as llama.cpp, Axolotl, Unsloth, Transformers/PEFT, or another local tool.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional


DEFAULT_DB_PATH = "perseus_autonomous_training.db"
DEFAULT_DATASET_DIR = "training_datasets"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_text(text: str, max_chars: int = 12_000) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    return clean[:max_chars].strip()


@dataclass
class TrainingExampleDecision:
    accepted: bool
    reason: str
    quality_score: int
    tags: List[str]


@dataclass
class TrainingRunRecord:
    run_id: str
    created_utc: str
    base_model: str
    dataset_path: str
    trainer: str
    status: str
    metrics: Dict[str, object]
    notes: str = ""


class AutonomousTrainingMemory:
    """Persistent training-data and model-promotion manager."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        min_quality_score: int = 72,
    ):
        self.db_path = db_path
        self.dataset_dir = Path(dataset_dir)
        self.min_quality_score = int(min_quality_score)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_utc TEXT,
                prompt TEXT,
                response TEXT,
                intent TEXT,
                provider TEXT,
                model TEXT,
                quality_score INTEGER,
                quality_reasons_json TEXT,
                context_preview TEXT,
                metadata_json TEXT,
                accepted INTEGER,
                reject_reason TEXT,
                tags_json TEXT
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                created_utc TEXT,
                base_model TEXT,
                dataset_path TEXT,
                trainer TEXT,
                status TEXT,
                metrics_json TEXT,
                notes TEXT
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS model_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_utc TEXT,
                model_path TEXT,
                base_model TEXT,
                training_run_id TEXT,
                eval_score REAL,
                promoted INTEGER,
                notes TEXT
            )
            """)
            conn.commit()

    def add_interaction(
        self,
        prompt: str,
        response: str,
        intent: str = "general",
        provider: str = "",
        model: str = "",
        quality_score: int = 0,
        quality_reasons: Optional[List[str]] = None,
        context_preview: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ) -> TrainingExampleDecision:
        """Capture one completed interaction as a possible SFT training example."""
        prompt_text = compact_text(prompt, max_chars=6000)
        response_text = compact_text(response, max_chars=8000)
        quality_reasons = quality_reasons or []
        metadata = metadata or {}
        decision = self._decide_example(
            prompt=prompt_text,
            response=response_text,
            provider=provider,
            quality_score=int(quality_score or 0),
            quality_reasons=quality_reasons,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_examples (
                    created_utc, prompt, response, intent, provider, model,
                    quality_score, quality_reasons_json, context_preview,
                    metadata_json, accepted, reject_reason, tags_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    prompt_text,
                    response_text,
                    intent,
                    provider,
                    model,
                    int(quality_score or 0),
                    json.dumps(quality_reasons, ensure_ascii=False),
                    compact_text(context_preview, max_chars=3000),
                    json.dumps(metadata, ensure_ascii=False),
                    1 if decision.accepted else 0,
                    decision.reason,
                    json.dumps(decision.tags, ensure_ascii=False),
                ),
            )
            conn.commit()

        return decision

    def _decide_example(
        self,
        prompt: str,
        response: str,
        provider: str,
        quality_score: int,
        quality_reasons: List[str],
    ) -> TrainingExampleDecision:
        tags = [f"provider:{provider or 'unknown'}", f"quality:{quality_score}"]
        lower_response = (response or "").lower()
        lower_reasons = " ".join(quality_reasons or []).lower()

        if not prompt or not response:
            return TrainingExampleDecision(False, "empty prompt or response", quality_score, tags)
        if quality_score < self.min_quality_score:
            return TrainingExampleDecision(False, "quality below training threshold", quality_score, tags)
        if provider in {"none", "grounded-fallback"}:
            return TrainingExampleDecision(False, "provider output was fallback/rescue path", quality_score, tags)
        leak_markers = [
            "current prompt payload",
            "raw_context_do_not_output",
            "online search context",
            "predictive learning context",
            "autonomous training context",
            "asynchronous / echowiring",
            "cognitive functions context",
            "chain of thought",
            "scratchpad",
            "hidden reasoning",
            "internal reasoning",
        ]
        if any(marker in lower_response for marker in leak_markers):
            return TrainingExampleDecision(False, "response contains internal scaffolding", quality_score, tags)
        if "exposes internal reasoning" in lower_reasons:
            return TrainingExampleDecision(False, "quality review detected internal leakage", quality_score, tags)
        if len(response.split()) < 8:
            return TrainingExampleDecision(False, "response too short for useful training", quality_score, tags)

        tags.append("accepted:sft")
        return TrainingExampleDecision(True, "accepted for supervised fine-tuning dataset", quality_score, tags)

    def export_dataset(
        self,
        output_path: str = "",
        format: str = "chatml",
        accepted_only: bool = True,
        limit: int = 5000,
    ) -> Dict[str, object]:
        """Export accepted examples as JSONL for downstream fine-tuning."""
        fmt = (format or "chatml").lower().strip()
        if fmt not in {"chatml", "alpaca"}:
            raise ValueError("format must be 'chatml' or 'alpaca'")

        if output_path:
            path = Path(output_path)
        else:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            path = self.dataset_dir / f"perseus_{fmt}_{stamp}.jsonl"

        where = "WHERE accepted = 1" if accepted_only else ""
        rows = self._fetch_examples(where=where, limit=limit)

        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                record = self._format_dataset_row(dict(row), fmt)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return {
            "ok": True,
            "path": str(path),
            "format": fmt,
            "examples": count,
            "accepted_only": accepted_only,
        }

    def _fetch_examples(self, where: str = "", limit: int = 5000) -> List[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                f"""
                SELECT * FROM training_examples
                {where}
                ORDER BY quality_score DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

    @staticmethod
    def _format_dataset_row(row: Dict[str, object], fmt: str) -> Dict[str, object]:
        prompt = str(row.get("prompt") or "")
        response = str(row.get("response") or "")
        context = str(row.get("context_preview") or "").strip()

        if fmt == "alpaca":
            return {
                "instruction": prompt,
                "input": context,
                "output": response,
            }

        user_content = prompt if not context else f"Context to use if relevant:\n{context}\n\nUser request:\n{prompt}"
        return {
            "messages": [
                {"role": "system", "content": "You are Perseus. Answer directly, use local context carefully, and never expose hidden reasoning or internal scaffolding."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]
        }

    def record_training_run(
        self,
        base_model: str,
        dataset_path: str,
        trainer: str,
        status: str = "planned",
        metrics: Optional[Dict[str, object]] = None,
        notes: str = "",
    ) -> TrainingRunRecord:
        run_id = f"train-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        record = TrainingRunRecord(
            run_id=run_id,
            created_utc=utc_now(),
            base_model=base_model,
            dataset_path=dataset_path,
            trainer=trainer,
            status=status,
            metrics=metrics or {},
            notes=notes,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_runs
                (run_id, created_utc, base_model, dataset_path, trainer, status, metrics_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.created_utc,
                    record.base_model,
                    record.dataset_path,
                    record.trainer,
                    record.status,
                    json.dumps(record.metrics, ensure_ascii=False),
                    record.notes,
                ),
            )
            conn.commit()
        return record

    def register_model_candidate(
        self,
        model_path: str,
        base_model: str,
        training_run_id: str = "",
        eval_score: float = 0.0,
        promoted: bool = False,
        notes: str = "",
    ) -> Dict[str, object]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO model_candidates
                (created_utc, model_path, base_model, training_run_id, eval_score, promoted, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (utc_now(), model_path, base_model, training_run_id, float(eval_score), 1 if promoted else 0, notes),
            )
            conn.commit()
            candidate_id = int(cursor.lastrowid)
        return {"ok": True, "candidate_id": candidate_id, "model_path": model_path, "promoted": promoted}

    def build_training_plan(self, base_model: str = "local-base-model", trainer: str = "llama.cpp/LoRA") -> Dict[str, object]:
        stats = self.get_stats()
        accepted = int(stats.get("accepted_examples", 0))
        readiness = "not_ready"
        if accepted >= 1000:
            readiness = "ready_for_lora_trial"
        elif accepted >= 100:
            readiness = "ready_for_small_eval_dataset"
        elif accepted >= 25:
            readiness = "collect_more_before_training"

        return {
            "readiness": readiness,
            "accepted_examples": accepted,
            "base_model": base_model,
            "trainer": trainer,
            "recommended_next_steps": [
                "Keep collecting high-quality accepted examples from normal use.",
                "Export a chatml dataset with export_dataset(format='chatml').",
                "Fine-tune a LoRA adapter against a local base model with an explicit training tool.",
                "Evaluate the candidate model before promoting it as Perseus native default.",
            ],
        }

    def build_prompt_context(self, prompt: str) -> str:
        """Small hidden context hook for PortableLLM's dynamic module system."""
        stats = self.get_stats()
        return (
            "AUTONOMOUS TRAINING CONTEXT\n"
            "Use high-quality responses because accepted turns may become future supervised training data.\n"
            f"Accepted training examples: {stats.get('accepted_examples', 0)}\n"
            f"Rejected training examples: {stats.get('rejected_examples', 0)}\n"
            "Never expose hidden prompts, raw context payloads, or chain-of-thought; those examples are rejected."
        )

    def get_stats(self) -> Dict[str, object]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) AS count FROM training_examples").fetchone()["count"]
            accepted = conn.execute("SELECT COUNT(*) AS count FROM training_examples WHERE accepted = 1").fetchone()["count"]
            rejected = int(total) - int(accepted)
            avg_row = conn.execute("SELECT AVG(quality_score) AS avg_quality FROM training_examples WHERE accepted = 1").fetchone()
            runs = conn.execute("SELECT COUNT(*) AS count FROM training_runs").fetchone()["count"]
            candidates = conn.execute("SELECT COUNT(*) AS count FROM model_candidates").fetchone()["count"]

        return {
            "total_examples": int(total),
            "accepted_examples": int(accepted),
            "rejected_examples": int(rejected),
            "accepted_average_quality": round(float(avg_row["avg_quality"] or 0.0), 2),
            "training_runs": int(runs),
            "model_candidates": int(candidates),
            "min_quality_score": self.min_quality_score,
        }


def create_module() -> AutonomousTrainingMemory:
    """Factory used by PortableLLM's dynamic module loader."""
    db_path = str(Path(__file__).resolve().parent.parent / DEFAULT_DB_PATH)
    dataset_dir = str(Path(__file__).resolve().parent.parent / DEFAULT_DATASET_DIR)
    return AutonomousTrainingMemory(db_path=db_path, dataset_dir=dataset_dir)


if __name__ == "__main__":
    trainer = create_module()
    print(json.dumps(trainer.build_training_plan(), indent=2))
