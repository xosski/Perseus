# Portable LLM Module

This folder provides a standard, portable LLM layer built using the existing Hades architecture.

## What It Uses

- `llm_conversation_core.ConversationManager` for provider management and conversation persistence
- `ollama` as the default strict local-only generation backend
- `offline_llm.OfflineLLM` as an optional fallback path only when explicitly enabled

## Response Sophistication

- Intent profiling for each prompt (`technical`, `educational`, `strategic`, `analytical`)
- Adaptive prompt contracts that enforce `Summary`, `Reasoning`, and `Next Steps`
- Response quality scoring that penalizes short, generic, unstructured, or low-reasoning answers
- Single-pass response refinement when initial output quality is low
- Strict local-only mode by default (no remote/fallback providers)
- Persistent self-improvement memory that learns from prior quality failures and injects corrective directives
- Runtime session metrics (`success`, `fallbacks`, `average_quality`, `refinements_used`)

## Web Learning Behavior

- News ingestion is feed-aware: it ingests both the feed source and linked article URLs
- Each feed is bounded to a fixed number of story links per run to keep ingestion practical
- Prompt enrichment pulls multiple relevant context snippets from learned web knowledge
- Grounded responses are expected to include an `Ingested Context Used` section when context is available
- Ingestion errors are handled per-source so one bad URL does not disable the learner for the whole session

## Quick Start

Run CLI chat:

```bash
python cli.py
```

Choose provider explicitly:

```bash
python cli.py --provider ollama --model llama3.2
```

Enable non-strict mode (allows fallbacks/remotes):

```bash
python cli.py --allow-fallbacks
```

Enable OfflineLLM fallback (only works with `--allow-fallbacks`):

```bash
python cli.py --allow-fallbacks --use-offline-fallback
```

## Python Usage

```python
from LLM import PortableLLM

llm = PortableLLM()
reply = llm.ask("Explain SQL injection prevention clearly.")
print(reply)

# Metadata includes provider, model, quality score, refinement status,
# and whether ingested web context was used.
reply, meta = llm.ask_with_metadata("Design a practical appsec roadmap for a startup team.")
print(meta)

print(llm.get_stats())
llm.close()
```

## Ingestion Notes

- The desktop UI has a `Web Ingest` tab for bulk source ingestion and manual URL ingestion
- Bulk ingest reports both feed success and linked article success counts
- Learned content is stored in `llm_web_learning.db`

## Portability Notes

- Defaults to strict local-only mode and requires a local provider (`ollama`)
- In strict mode, remote/fallback providers are blocked and provider switching is limited to local providers
- If local provider output is weak, one refinement pass is attempted before returning best local result
- Optional non-strict behavior is available via CLI flags when you intentionally want fallback behavior
- Stores chat history in `llm_portable_conversations.db` by default
- Stores self-improvement episodes in `llm_self_improvement.db` to guide future prompting
