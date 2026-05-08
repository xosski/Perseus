# Portable LLM Module

This folder provides a standard, portable LLM layer built using the existing Hades architecture.

## What It Uses

- `llm_conversation_core.ConversationManager` for provider management and conversation persistence
- `offline_llm.OfflineLLM` for smart local fallback responses

## Response Sophistication

- Intent profiling for each prompt (`technical`, `educational`, `strategic`, `analytical`)
- Adaptive prompt contracts that enforce `Summary`, `Reasoning`, and `Next Steps`
- Response quality scoring that penalizes short, generic, unstructured, or low-reasoning answers
- Single-pass response refinement when initial output quality is low
- Provider failover chain before local offline fallback
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

- Defaults to the best available provider in this order: `ollama`, `openai`, `mistral`, `azure`, `fallback`
- If a provider response is weak, one refinement pass is attempted before failover
- If no provider returns a usable answer, it uses `OfflineLLM` for educated/smart local responses
- Stores chat history in `llm_portable_conversations.db` by default
