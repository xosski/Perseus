# Perseus Portable LLM

Perseus is a local-first LLM workspace built around `portable_llm.py`, `cli.py`, and the dynamically loaded modules in `Modules/`. It prefers local execution through Ollama, learned local knowledge, and persistent memory databases before using any optional fallback path.

## What It Uses

- `portable_llm.PortableLLM` as the main orchestration layer
- `llm_conversation_core.ConversationManager` for persisted conversations, with a built-in fallback manager when needed
- Ollama as the default local generation backend
- Built-in local knowledge and grounded fallback responses when a generative model is not available
- `offline_llm.OfflineLLM` only as an optional non-strict fallback when installed and explicitly enabled
- Online search augmentation for explicit/current-information requests, while keeping model execution local by default
- A dynamic `Modules/` loader that imports Python-compatible `.py`, `.txt`, and extensionless module files without requiring normal package names

## Updated Module Stack

Perseus now includes a broader set of modules under `Modules/`. These modules are loaded at startup when compatible and either contribute hidden prompt context, learning memory, response repair, or provider/search support.

| Module | Main class/functions | Purpose |
| --- | --- | --- |
| `Brain State.py` | `BrainStateEngine` | Deterministic cognitive control state: attention, uncertainty, active goal, focus terms, response strategy, and post-response lessons. |
| `English Language.py` | `EnglishLanguageModule` | Pre-response English comprehension pass for task type, ambiguity, constraints, tone, entities, and expected answer shape. |
| `Coding Module.py` | `CodingModule`, `build_prompt_context()`, `analyze()` | Coding-assistant context for debug/refactor/create/review/security/test requests, plus local coding lesson storage. |
| `Search Augmentation.py` | `SearchAugmentation` | Decides when current online information is needed, queries configured providers/fallbacks, and caches search results in SQLite. |
| `Predictive learning.py` | `PredictiveLearningMemory` | Stores events, patterns, outcomes, and lessons so future prompts can retrieve relevant predictive context. |
| `Asyncronous Learning.py` | `EchoWiringMemory` | Predictive/asynchronous memory layer with AMM/EchoWiring fields and consent safeguards. |
| `Cognitive Functions.py` | `GhostCoreCognitiveEngine` | Inspectable cognitive-memory layer that stores memory traces, claims, assumptions, risks, and self-model updates. |
| `cognitive_memory.py` | `CognitiveLayer`, `MemoryAnalyzer` | Hades-compatible durable vector memory that compares recent chat focus with recalled long-term context. |
| `environment_awareness.py` | `EnvironmentObserver` | Bounded, read-only project/runtime snapshots used by active cognition before planning. |
| `Introspective Learning.py` | `IntrospectiveLearning` | Post-response critique and repair layer that catches weak answers, scaffold leakage, and internal reasoning exposure. |
| `response_rationale.py` | `ResponseRationale` | Persists inspectable evidence for why a response was selected: active context channels, contributing modules, provider/model, quality, strategy, and repairs. It does not expose or claim private chain-of-thought. |
| `self_code_module.py` | `SelfCodeModule` | Optional workspace-confined, staged self-code updates. Excluded from startup loading and enabled only after explicit user consent; every apply requires a second approval. |
| `Autonomous Training.py` | `AutonomousTrainingMemory` | Captures high-quality interactions as candidate supervised-training examples and exports clean JSONL datasets. |
| `Monday personality.py` | `build_monday_prompt()` | Optional Monday-style personality prompt builder. |
| `Monday Cook.py` | `build_monday_prompt()`, `build_task_wrapper()` | Alternate Monday prompt/wrapper helpers and example message formatting. |
| `Perseus_Memory_Orchestrator.py` | `PerseusMemoryOrchestrator` | Reference/optional local-first orchestration wrapper for brain state, EchoWiring, introspection, and autonomous training. |

### Module Persistence

The modules store their local state in SQLite databases beside the code by default:

- `llm_portable_conversations.db` - conversations plus durable Hades-compatible cognitive memories and reflections
- `llm_web_learning.db` - ingested web/local/chat knowledge
- `llm_search_cache.db` - online search cache
- `llm_self_improvement.db` - self-improvement episodes
- `brain_state_memory.db` - deterministic brain-state snapshots/transitions
- `predictive_learning_memory.db` - predictive learning events and patterns
- `ghostcore_echowiring_memory.db` - EchoWiring/asynchronous memory
- `ghostcore_cognitive_state.db` - cognitive traces and snapshots
- `introspective_learning.db` - critique/repair traces
- `perseus_response_rationale.db` - observable response-selection evidence and hashes
- `perseus_autonomous_training.db` - accepted/rejected training examples and model-candidate metadata

### Three Memory Channels

Perseus cognitive memory now uses logical channels with one canonical writer:

```text
ACTIVE cognition ── proposals ──► MEMORY coordinator ◄── proposals ── REFLECTION
       │                              │
       └──────── reads ◄──────────────┤
                                      ▼
                            append-only event journal
```

- **Active cognition** recalls canonical snapshots and proposes `ADD`, `UPDATE`, `ACCESS`, `REINFORCE`, or `FORGET` mutations.
- **Memory authority** serializes proposals through one queue, checks `expected_revision`, updates the current snapshot, and appends an audit event. Stale writes become reconciliation conflicts instead of overwrites.
- **Reflection** analyzes stable snapshots and submits decay/prune proposals to the same authority; it cannot directly change the store.

Each canonical memory includes `revision`, `created_at`, `updated_at`, `source_thread`, `confidence`, `parent_revision`, and status. Forgotten memories remain in the revision/event history instead of being physically erased.

## Response Sophistication

- Intent profiling for each prompt (`technical`, `educational`, `strategic`, `analytical`, `feedback`)
- English-language pre-analysis before retrieval and generation
- Deterministic brain-state planning that tracks active goal, focus terms, uncertainty policy, and response strategy
- Read-only awareness of the local project root, runtime, Git branch/status, and top-level structure before brain-state planning
- Question decomposition that privately evaluates relevant `who`, `what`, `when`, `where`, `why`, and `how` dimensions before answering
- Dynamic module context injection from compatible `Modules/` files
- Predictive, asynchronous, and cognitive memory lookup before generation
- Short-term versus long-term memory analysis that keeps the latest request authoritative while blending in relevant durable context
- Adaptive prompt contracts that favor candid, genuine feedback with clear rationale and next steps without exposing chain-of-thought
- Response quality scoring that penalizes short, generic, flattering, unstructured, low-rationale, or hidden-reasoning-leak answers
- Introspective post-response repair when a draft is weak, irrelevant, or leaks internal scaffolding
- Single-pass response refinement when initial output quality is low
- Visible-output sanitization that strips private reasoning tags, scratchpad sections, and hidden planning before responses are stored, learned from, or returned
- Strict local-only mode by default: local fallback first, then Ollama as a rescue provider, with remote providers blocked unless non-strict mode is enabled
- Persistent self-improvement memory that learns from prior quality failures and injects corrective directives
- Runtime metrics for provider behavior, quality, fallbacks, refinements, module status, and autonomous-training capture
- A response rationale for each successful answer, available with `/why`, the desktop **Why?** button, response metadata, or `get_last_response_rationale()`

The response rationale explains the recorded pipeline evidence that influenced selection. It deliberately does not present generated chain-of-thought as ground truth or claim complete access to model internals.

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

Choose provider explicitly:

```bash
python cli.py --provider ollama --model llama3.3
```

You can also set default model overrides with environment variables such as `PERSEUS_OLLAMA_MODEL`, `PERSEUS_OPENAI_MODEL`, `PERSEUS_MISTRAL_MODEL`, and `PERSEUS_AZURE_DEPLOYMENT`.

Use a custom conversation database:

```bash
python cli.py --db llm_portable_conversations.db
```

Disable online search augmentation for a fully offline session:

```bash
python cli.py --no-online-search
```

Disable durable learning from chat turns while still allowing manual knowledge ingestion:

```bash
python cli.py --no-chat-learning
```

Print the runtime safety/privacy/transparency report and exit:

```bash
python cli.py --compliance-report
```

Print local growth-learning stats, including distilled memories, lessons, contradictions, and the last benchmark run:

```bash
python cli.py --growth-report
```

Replay stored chat memories into compact confidence-scored lessons:

```bash
python cli.py --experience-replay 80
```

Run the local self-evaluation benchmark suite:

```bash
python cli.py --run-benchmarks 3
```

Disable startup folder ingestion:

```bash
python cli.py --no-auto-ingest-folders
```

Add one or more startup knowledge folders:

```bash
python cli.py --knowledge-folder knowledge --knowledge-folder "Princess protocol"
```

Enable non-strict mode, which allows optional fallback behavior:

```bash
python cli.py --allow-fallbacks
```

Enable OfflineLLM fallback. This only works with `--allow-fallbacks`:

```bash
python cli.py --allow-fallbacks --use-offline-fallback
```

## Ollama Setup

Perseus is designed to use Ollama as the default local LLM provider. Ollama must be running and at least one model must be installed. If Ollama is running but has no matching model, Perseus falls back to its local non-generative response paths.

1. Install Ollama from <https://ollama.com/download>.
2. Start the Ollama server:

```bash
ollama serve
```

3. In a second terminal, install a model:

```bash
ollama pull llama3.3
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
python cli.py --terminal --provider ollama --model llama3.3
```

or, for the smaller model:

```bash
python cli.py --terminal --provider ollama --model qwen2.5:1.5b
```

### Ollama Troubleshooting

- `Provider=fallback` means Perseus did not find a usable local Ollama model or used the local deterministic path first.
- If the requested Ollama model is not installed, Perseus now picks a compatible installed local model when possible instead of immediately failing.
- `HTTP Error 404` from Ollama usually means the requested model name is not installed and no compatible local fallback was found. Run `ollama list`, then pass the exact installed model name with `--model`.
- If `ollama list` is empty, install a model with `ollama pull llama3.3` or `ollama pull qwen2.5:1.5b`.
- If Ollama is not reachable, start it with `ollama serve`.
- In strict local-only mode, remote/cloud providers are blocked, so a missing Ollama model will use the built-in fallback instead of an online LLM.
- The built-in fallback can answer simple chat and some grounded learned-knowledge questions, but full ChatGPT-style arbitrary generation requires an installed Ollama model.

### Ingestion Hardening

- URL ingestion only accepts public `http://` and `https://` targets.
- Localhost, private-network, link-local, multicast, reserved, and oversized responses are rejected.
- Fetch timeouts and provider generation token limits are clamped to bounded ranges.

## Python Usage

```python
from Perseus import PortableLLM

llm = PortableLLM()
reply = llm.ask("Explain SQL injection prevention clearly.")
print(reply)

# Metadata includes provider, model, quality score, refinement status,
# online-search usage, brain-state metadata, module status, and whether
# ingested web/local context was used.
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

# Inspect dynamic modules, runtime statistics, and the latest response rationale.
print(llm.list_loaded_modules())
print(llm.get_stats())
print(llm.get_last_response_rationale())

# Self-code is OFF by default. A host must obtain user consent before enabling it.
print(llm.enable_self_code(user_approved=True))
proposal = llm.stage_self_update(
    "Modules/example_module.py",
    "VALUE = 1\n",
    rationale="Add an approved local module",
)
print(proposal["diff"])
# Ask the user to review the diff, then make a separate approval call.
result = llm.apply_self_update(proposal["proposal_id"], user_approved=True)
print(result)

# Export module-managed state when available.
print(llm.export_brain_state())
print(llm.export_training_dataset(format="chatml", limit=5000))

llm.close()
```

If running from inside this folder instead of importing the package from its parent, use:

```python
from portable_llm import PortableLLM
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

## Module Development Notes

- Place dynamic extensions in `Modules/`.
- `.py`, Python-compatible `.txt`, and extensionless files can be loaded.
- Files and folders such as `__pycache__`, `.git`, `.venv`, `venv`, `env`, and `node_modules` are skipped.
- A module can contribute hidden prompt context by exposing a compatible object/function such as `MODULE_INSTANCE`, `build_prompt_context(prompt)`, `get_prompt_context(prompt)`, `get_context(prompt)`, or `analyze(prompt)`.
- Module context is treated as hidden scaffolding. User-visible responses should not mention raw module names, hidden context blocks, chain-of-thought, scratchpads, or internal reasoning unless the user explicitly asks to inspect modules.
- `self_code_module.py` is a deliberate exception to normal startup scanning. The desktop and terminal launchers ask after startup whether to load it for that session. Declining leaves it unimported.
- Enabling self-code does not authorize writes. Updates are workspace-relative, extension-limited, syntax-checked, staged as unified diffs, and applied only through a separate explicit approval; existing files are backed up first. Python changes require a restart before the running process uses them.
- `Perseus_Memory_Orchestrator.py` is useful as a reference/optional wrapper, but `portable_llm.py` already performs the main orchestration internally.

## Safety, Privacy, and Transparency Notes

- `--compliance-report` prints a runtime self-assessment of local-only mode, chat-learning state, online-search state, sanitizer/repair controls, audit databases, and loaded modules.
- The compliance report is a transparency aid, not a legal certification against every AI standard.
- `--no-chat-learning` disables durable learning from chat turns into the web learner, predictive memory, EchoWiring memory, cognitive memory, and autonomous-training capture.
- `--no-online-search` disables online search augmentation for sessions that must remain fully offline.
- Strict local-only mode blocks remote providers unless `--allow-fallbacks` is explicitly used.
- Response rationale records observable pipeline evidence and content hashes, not raw prompt/response copies or private chain-of-thought.
- The self-code module is excluded from startup import, requires session opt-in, and requires separate approval for every staged filesystem change.

## Portability Notes

- Defaults to strict local-only mode and prefers local execution
- Online search is enabled by default only for explicit or current-information lookup requests; use `--no-online-search` to disable it
- If a generative model is not active, the built-in local path keeps the app centered on ingested knowledge instead of web browsing
- In strict mode, remote providers are blocked and provider switching is limited to local providers
- If local provider output is weak, introspective repair and one refinement pass may be attempted before returning the best local result
- Optional non-strict behavior is available via CLI flags when you intentionally want fallback behavior
- Stores chat history in `llm_portable_conversations.db` by default
- Stores self-improvement episodes in `llm_self_improvement.db` to guide future prompting
- Stores module-specific learning state in the SQLite databases listed above
