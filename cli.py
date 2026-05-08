"""Portable LLM desktop chat launcher."""

from __future__ import annotations

import argparse

try:
    from .portable_llm import launch_portable_llm_chat
except ImportError:
    from portable_llm import launch_portable_llm_chat


def run() -> int:
    parser = argparse.ArgumentParser(description="Portable LLM chat window launcher")
    parser.add_argument("--provider", default=None, help="Provider name (strict mode only allows ollama)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--db", default="llm_portable_conversations.db", help="Conversation database path")
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Disable strict local-only mode and allow remote/fallback providers",
    )
    parser.add_argument(
        "--use-offline-fallback",
        action="store_true",
        help="Enable OfflineLLM fallback (only used when --allow-fallbacks is set)",
    )
    args = parser.parse_args()

    launch_portable_llm_chat(
        db_path=args.db,
        provider=args.provider,
        model=args.model,
        strict_local_only=not args.allow_fallbacks,
        use_offline_fallback=args.use_offline_fallback,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
