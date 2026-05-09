# Portable LLM Module

This folder provides a standard, portable LLM layer built using the existing Hades architecture.

## What It Uses

- `llm_conversation_core.ConversationManager` when available, with a built-in local conversation manager fallback
- `ollama` as the default strict local-only generation backend
- A built-in local knowledge response path that keeps the app focused on learned sources when a generative model is not active
- `offline_llm.OfflineLLM` as an optional enhanced fallback path only when installed and explicitly enabled

## Response Sophistication

- Intent profiling for each prompt (`technical`, `educational`, `strategic`, `analytical`, `feedback`)
- Question decomposition that reasons through relevant `who`, `what`, `when`, `where`, `why`, and `how` dimensions before answering
- Adaptive prompt contracts that favor candid, genuine feedback with clear reasoning and next steps
- Response quality scoring that penalizes short, generic, flattering, unstructured, or low-reasoning answers
- Single-pass response refinement when initial output quality is low
- Strict local-only mode by default (Ollama first, learned-source knowledge response when needed)
- Persistent self-improvement memory that learns from prior quality failures and injects corrective directives
- Runtime session metrics (`success`, `fallbacks`, `average_quality`, `refinements_used`)

## Knowledge Learning Behavior

- Source-site ingestion accepts RSS/Atom feeds and normal websites in one persistent list
- Feeds ingest both the feed source and linked article URLs
- Normal websites ingest the page plus a bounded set of same-site links so Perseus can learn from trusted sources without manual browsing
- Each source is bounded to keep ingestion practical
- Local folder ingestion reads supported text files from a knowledge folder into the same learning store
- Chat learning stores useful user inputs and completed exchanges as local memory so future answers can adapt to the user's facts, preferences, and project context
- Prompt enrichment pulls multiple relevant context snippets from learned web knowledge
- Grounded responses are expected to include an `Ingested Context Used` section when context is available
- Ingestion errors are handled per-source so one bad URL does not disable the learner for the whole session

## Quick Start

Run desktop chat:

```bash
python cli.py
```

Run terminal chat:

```bash
python cli.py --terminal
```

## Ollama Setup

Perseus is designed to use Ollama as the default local LLM provider. Ollama must be running **and** at least one model must be installed. If Ollama is running but has no models, Perseus will fall back to its basic local fallback responses.

1. Install Ollama from <https://ollama.com/download>.
2. Start the Ollama server:

```bash
ollama serve
```

3. In a second terminal, install a model:

```bash
ollama pull llama3.2
```

If your machine is low on RAM, try a smaller model:

```bash
ollama pull qwen2.5:1.5b
```

4. Confirm Ollama sees the model:

```bash
ollama list
```

5. Launch Perseus with the matching model:

```bash
python cli.py --terminal --provider ollama --model llama3.2
```

or, for the smaller model:

```bash
python cli.py --terminal --provider ollama --model qwen2.5:1.5b
```

### Ollama Troubleshooting

- `Provider=fallback` means Perseus did not find a usable local Ollama model.
- `HTTP Error 404` from Ollama usually means the requested model name is not installed. Run `ollama list`, then pass the exact installed model name with `--model`.
- If `ollama list` is empty, install a model with `ollama pull llama3.2` or `ollama pull qwen2.5:1.5b`.
- If Ollama is not reachable, start it with `ollama serve`.
- In strict local-only mode, remote/cloud providers are blocked, so a missing Ollama model will use the built-in fallback instead of an online LLM.
- The built-in fallback can answer simple chat and some basic general questions, but full ChatGPT-style arbitrary responses require an installed Ollama model.

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

# Ingest local knowledge files. By default this expects a `knowledge/` folder
# beside this module, but you can pass any folder path.
ingest_result = llm.ingest_folder("knowledge")
print(ingest_result)

# Persist and ingest trusted source sites/feeds.
llm.save_source_sites([
    "https://docs.python.org/3/",
    "https://krebsonsecurity.com/feed/",
])
source_result = llm.ingest_source_sites()
print(source_result)

print(llm.get_stats())
llm.close()
```

## Ingestion Notes

- The desktop UI has a `Knowledge Ingest` tab for source-site ingestion, manual URL ingestion, and local folder ingestion
- The source-site list is stored in `knowledge_sources.json`; edit it in the UI or directly as JSON
- Source sites can be RSS/Atom feeds or normal websites; normal websites are scanned for a bounded number of same-site links
- The default startup local folders are `knowledge` and `Princess protocol`; use `--knowledge-folder` to override or `--no-auto-ingest-folders` to disable startup folder ingestion
- Folder ingestion is recursive by default and supports `.txt`, `.md`, `.docx`, `.py`, `.json`, `.yaml`, `.yml`, `.csv`, `.html`, `.htm`, `.log`, `.typed`, and no-extension text files
- Individual local files larger than 1 MB are skipped to keep ingestion responsive
- User chat turns are learned into the same `llm_web_learning.db` store as `perseus://chat-memory/...` records
- Source ingest reports both source success and linked page/article success counts
- Learned content is stored in `llm_web_learning.db`

## Portability Notes

- Defaults to strict local-only mode and prefers a local provider (`ollama`)
- If a generative model is not active, the built-in local path keeps the app centered on ingested knowledge instead of web browsing
- In strict mode, remote providers are blocked and provider switching is limited to local providers
- If local provider output is weak, one refinement pass is attempted before returning best local result
- Optional non-strict behavior is available via CLI flags when you intentionally want fallback behavior
- Stores chat history in `llm_portable_conversations.db` by default
- Stores self-improvement episodes in `llm_self_improvement.db` to guide future prompting
