"""Portable LLM desktop chat launcher."""

from __future__ import annotations

import argparse

try:
    from .portable_llm import launch_portable_llm_chat
except ImportError:
    from portable_llm import launch_portable_llm_chat


def run() -> int:
    parser = argparse.ArgumentParser(description="Portable LLM chat window launcher")
    parser.add_argument("--provider", default=None, help="Provider name (ollama/openai/mistral/azure/fallback)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--db", default="llm_portable_conversations.db", help="Conversation database path")
    args = parser.parse_args()

    launch_portable_llm_chat(db_path=args.db, provider=args.provider, model=args.model)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
