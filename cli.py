"""Portable LLM desktop chat launcher."""

from __future__ import annotations

import argparse
import json

try:
    from .portable_llm import PortableLLM, launch_portable_llm_chat, launch_portable_llm_terminal_chat
except ImportError:
    from portable_llm import PortableLLM, launch_portable_llm_chat, launch_portable_llm_terminal_chat


def run() -> int:
    parser = argparse.ArgumentParser(description="Portable LLM chat window launcher")
    parser.add_argument("--provider", default=None, help="Provider name (strict mode only allows ollama)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--db", default="llm_portable_conversations.db", help="Conversation database path")
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Run a ChatGPT-style terminal chat instead of the desktop window",
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Disable strict local-only mode and allow remote/fallback providers",
    )
    parser.add_argument(
        "--no-online-search",
        action="store_true",
        help="Disable online web search augmentation while keeping other features enabled",
    )
    parser.add_argument(
        "--no-chat-learning",
        action="store_true",
        help="Disable durable learning from chat turns while keeping manual knowledge ingestion available",
    )
    parser.add_argument(
        "--use-offline-fallback",
        action="store_true",
        help="Enable OfflineLLM fallback (only used when --allow-fallbacks is set)",
    )
    parser.add_argument(
        "--knowledge-folder",
        action="append",
        default=None,
        help=(
            "Local knowledge folder to auto-ingest on startup. "
            "Can be repeated; defaults to knowledge and Princess protocol."
        ),
    )
    parser.add_argument(
        "--no-auto-ingest-folders",
        action="store_true",
        help="Disable startup auto-ingestion of local knowledge folders",
    )
    parser.add_argument(
        "--compliance-report",
        action="store_true",
        help="Print Perseus runtime safety/privacy/transparency controls and exit",
    )
    args = parser.parse_args()

    if args.compliance_report:
        llm = PortableLLM(
            db_path=args.db,
            provider=args.provider,
            model=args.model,
            strict_local_only=not args.allow_fallbacks,
            allow_online_search=not args.no_online_search,
            use_offline_fallback=args.use_offline_fallback,
            enable_chat_learning=not args.no_chat_learning,
        )
        try:
            print(json.dumps(llm.get_compliance_report(), indent=2))
        finally:
            llm.close()
        return 0

    launcher = launch_portable_llm_terminal_chat if args.terminal else launch_portable_llm_chat
    launcher(
        db_path=args.db,
        provider=args.provider,
        model=args.model,
        strict_local_only=not args.allow_fallbacks,
        allow_online_search=not args.no_online_search,
        use_offline_fallback=args.use_offline_fallback,
        enable_chat_learning=not args.no_chat_learning,
        knowledge_folders=args.knowledge_folder,
        auto_ingest_folders=not args.no_auto_ingest_folders,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
