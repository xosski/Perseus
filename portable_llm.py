"""
Portable LLM orchestrator.

Uses the existing architecture:
- llm_conversation_core.ConversationManager for provider routing and persistence
- offline_llm.OfflineLLM for smart offline fallback

Enhancements in this module:
- Prompt intent profiling (technical, educational, strategic, analytical)
- Adaptive system prompting contracts
- Lightweight response quality scoring
- Single-pass refinement for weak drafts
- Multi-provider failover before offline fallback
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from html import unescape
import json
import logging
from pathlib import Path
import queue
import re
import sqlite3
import threading
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

try:
    from llm_conversation_core import ConversationManager
    from offline_llm import OfflineLLM
except ImportError:
    @dataclass
    class ConversationMessage:
        role: str
        content: str
        metadata: Dict[str, object]


    @dataclass
    class ConversationSession:
        id: int
        title: str
        provider: str
        model: str
        system_prompt: str
        temperature: float = 0.7
        max_tokens: int = 2048
        messages: List[ConversationMessage] = None

        def __post_init__(self) -> None:
            if self.messages is None:
                self.messages = []

        def add_message(self, role: str, content: str, metadata: Optional[Dict[str, object]] = None) -> None:
            self.messages.append(ConversationMessage(role=role, content=content, metadata=metadata or {}))


    class LocalOllamaProvider:
        name = "ollama"

        def __init__(self) -> None:
            self.available = self._is_available()

        @staticmethod
        def _is_available() -> bool:
            try:
                req = Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "Perseus/1.0"})
                with urlopen(req, timeout=2) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                return bool(payload.get("models"))
            except Exception:
                return False

        def generate(
            self,
            prompt: str,
            messages: Optional[List[Dict[str, str]]] = None,
            model: str = "llama3.2",
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> Optional[str]:
            payload = {
                "model": model,
                "messages": messages or [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192},
            }
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                "http://127.0.0.1:11434/api/chat",
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "Perseus/1.0"},
                method="POST",
            )
            with urlopen(req, timeout=180) as resp:
                parsed = json.loads(resp.read().decode("utf-8", errors="replace"))
            message = parsed.get("message") or {}
            return (message.get("content") or "").strip() or None


    class LocalFallbackProvider:
        name = "fallback"
        available = True

        def generate(
            self,
            prompt: str,
            messages: Optional[List[Dict[str, str]]] = None,
            model: str = "fallback",
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> str:
            user_prompt = _extract_user_request(prompt)
            prompt_lower = user_prompt.lower()

            if _is_capability_prompt(user_prompt):
                return _capability_response()

            general_answer = _general_knowledge_fallback(user_prompt)
            if general_answer:
                return general_answer

            if _is_general_knowledge_prompt(user_prompt):
                return _heuristic_general_answer(user_prompt)

            if _is_small_talk_prompt(user_prompt):
                if any(token in prompt_lower for token in ["how are you", "how's it going", "how is it going"]):
                    return (
                        "I'm doing alright - alert, local, and mildly annoyed that Ollama is not available, "
                        "but still ready to help. Tell me what we're working on today."
                    )
                if any(token in prompt_lower for token in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
                    return "Hey - I'm here. Ask me anything, or point me at a file/folder and I'll work from it."
                if "thank" in prompt_lower:
                    return "You're welcome. Send the next thing when you're ready."

                return "I'm here and ready. What would you like to dig into?"

            return (
                "Knowledge Response:\n"
                "I do not have enough learned context to answer that with confidence yet. "
                "Add trusted sites, feeds, URLs, or local files in Knowledge Ingest, then ask again and I will ground the answer in that material.\n\n"
                "What to add:\n"
                "- Primary documentation or official sources for the topic.\n"
                "- High-signal news, security, reference, or research sources you trust.\n"
                "- Local notes, project docs, or personal reference files.\n\n"
                f"Request to ground: {user_prompt}"
            )


    class ConversationManager:
        """Small local replacement when llm_conversation_core is not installed."""

        def __init__(self, db_path: str = "llm_portable_conversations.db"):
            self.db_path = db_path
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._lock = threading.Lock()
            self.providers = {
                "ollama": LocalOllamaProvider(),
                "fallback": LocalFallbackProvider(),
            }
            self._initialize()

        def _initialize(self) -> None:
            with self.conn:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT,
                        provider TEXT,
                        model TEXT,
                        system_prompt TEXT,
                        temperature REAL,
                        max_tokens INTEGER,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id INTEGER,
                        role TEXT,
                        content TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

        def get_available_providers(self) -> List[str]:
            self.providers["ollama"].available = self.providers["ollama"]._is_available()
            return [name for name, provider in self.providers.items() if provider.available]

        def create_conversation(self, title: str, provider: str, model: str, system_prompt: str) -> ConversationSession:
            with self._lock:
                with self.conn:
                    cur = self.conn.execute(
                        """
                        INSERT INTO conversations (title, provider, model, system_prompt, temperature, max_tokens)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (title, provider, model, system_prompt, 0.7, 2048),
                    )
                return ConversationSession(
                    id=int(cur.lastrowid),
                    title=title,
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                )

        def _save_conversation(self, conversation: ConversationSession) -> None:
            with self._lock:
                with self.conn:
                    self.conn.execute(
                        """
                        UPDATE conversations
                        SET provider = ?, model = ?, system_prompt = ?, temperature = ?, max_tokens = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (
                            conversation.provider,
                            conversation.model,
                            conversation.system_prompt,
                            conversation.temperature,
                            conversation.max_tokens,
                            conversation.id,
                        ),
                    )
                    self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation.id,))
                    for message in conversation.messages:
                        self.conn.execute(
                            """
                            INSERT INTO messages (conversation_id, role, content, metadata)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                conversation.id,
                                message.role,
                                message.content,
                                json.dumps(message.metadata),
                            ),
                        )

        def switch_provider(self, conversation_id: int, provider: str, model: str) -> bool:
            if provider not in self.providers or not self.providers[provider].available:
                return False
            with self._lock:
                with self.conn:
                    self.conn.execute(
                        "UPDATE conversations SET provider = ?, model = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (provider, model, conversation_id),
                    )
            return True


    class OfflineLLM:
        """Minimal offline fallback used when the external OfflineLLM module is unavailable."""

        def __init__(self, use_knowledge_db: bool = True):
            self.use_knowledge_db = use_knowledge_db

        def generate(self, user_input: str, mood: str = "pragmatic", system_prompt: str = "") -> str:
            return LocalFallbackProvider().generate(prompt=user_input)

        def close(self) -> None:
            return None

try:
    from web_knowledge_learner import WebKnowledgeLearner
except ImportError:
    class LocalKnowledgeStore:
        def __init__(self, conn: sqlite3.Connection):
            self.conn = conn

        def get_learning_stats(self) -> Dict[str, int]:
            row = self.conn.execute("SELECT COUNT(*) AS count FROM learned_documents").fetchone()
            return {"documents": int(row["count"] if row else 0)}


    class WebKnowledgeLearner:
        """Small local text knowledge store when web_knowledge_learner is not installed."""

        def __init__(self, db_path: str = "llm_web_learning.db"):
            self.db_path = db_path
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._lock = threading.Lock()
            self._initialize()
            self.store = LocalKnowledgeStore(self.conn)

        def _initialize(self) -> None:
            with self.conn:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS learned_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT UNIQUE,
                        title TEXT,
                        content TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

        def learn_from_content(self, url: str, content: str, metadata: Optional[Dict[str, object]] = None) -> Dict[str, object]:
            title = str((metadata or {}).get("title") or url)
            text = (content or "").strip()
            if not text:
                return {"total_items_learned": 0, "reason": "empty content"}
            with self._lock:
                with self.conn:
                    self.conn.execute(
                        """
                        INSERT INTO learned_documents (url, title, content)
                        VALUES (?, ?, ?)
                        ON CONFLICT(url) DO UPDATE SET
                            title = excluded.title,
                            content = excluded.content,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (url, title, text),
                    )
            return {"total_items_learned": 1, "title": title, "chars": len(text)}

        def get_knowledge_context_for_query(self, query: str, limit: int = 3) -> str:
            terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_-]{3,}", query or "")[:8]]
            if not terms:
                return ""

            rows = self.conn.execute(
                """
                SELECT title, content
                FROM learned_documents
                ORDER BY updated_at DESC, id DESC
                LIMIT 200
                """
            ).fetchall()

            scored: List[Tuple[int, sqlite3.Row]] = []
            for row in rows:
                haystack = f"{row['title']}\n{row['content']}".lower()
                score = sum(haystack.count(term) for term in terms)
                if score > 0:
                    scored.append((score, row))

            snippets: List[str] = []
            for _score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]:
                content = re.sub(r"\s+", " ", row["content"]).strip()[:900]
                snippets.append(f"Source: {row['title']}\n{content}")

            return "\n\n".join(snippets)

        def get_all_knowledge_context(self, limit: int = 8) -> str:
            rows = self.conn.execute(
                """
                SELECT title, content
                FROM learned_documents
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

            snippets: List[str] = []
            for row in rows:
                content = re.sub(r"\s+", " ", row["content"]).strip()[:900]
                snippets.append(f"Source: {row['title']}\n{content}")

            return "\n\n".join(snippets)

        def close(self) -> None:
            self.conn.close()

WEB_LEARNING_DB_PATH = "llm_web_learning.db"
SELF_IMPROVEMENT_DB_PATH = "llm_self_improvement.db"
MONDAY_PERSONALITY_FILE = "MOnday personality.txt"
OLLAMA_SMART_CONTENT_FILE = "Ollama smart content.txt"
DEFAULT_KNOWLEDGE_FOLDER = "knowledge"
DEFAULT_PRINCESS_PROTOCOL_FOLDER = "Princess protocol"
DEFAULT_AUTO_KNOWLEDGE_FOLDERS = (DEFAULT_KNOWLEDGE_FOLDER, DEFAULT_PRINCESS_PROTOCOL_FOLDER)
SUPPORTED_KNOWLEDGE_EXTENSIONS = {
    ".txt",
    ".md",
    ".docx",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".html",
    ".htm",
    ".log",
    ".typed",
    "",
}
MAX_KNOWLEDGE_FILE_BYTES = 1_000_000

logger = logging.getLogger("PortableLLM")


@dataclass
class PromptProfile:
    """Intent profile used to adapt prompting and evaluation."""

    intent: str
    complexity: str
    mood: str
    expected_shape: str
    prefer_structure: bool
    prefer_concise: bool
    conversational: bool


@dataclass
class ResponseQuality:
    """Heuristic quality score for generated responses."""

    score: int
    reasons: List[str]


@dataclass
class LLMStats:
    """Operational stats for the running session."""

    total_requests: int = 0
    provider_successes: int = 0
    provider_failures: int = 0
    refinements_used: int = 0
    offline_fallbacks: int = 0
    average_quality: float = 0.0


@dataclass
class EnrichedPrompt:
    """Prompt payload enriched with learned context metadata."""

    text: str
    has_context: bool
    context_preview: str = ""


class SelfImprovementStore:
    """Persist lightweight generation outcomes so prompting can improve over time."""

    def __init__(self, db_path: str = SELF_IMPROVEMENT_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS improvement_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    intent TEXT,
                    quality_score INTEGER,
                    quality_reasons TEXT,
                    was_refined INTEGER,
                    used_context INTEGER,
                    response_chars INTEGER
                )
                """
            )

    def record(
        self,
        intent: str,
        quality_score: int,
        quality_reasons: List[str],
        was_refined: bool,
        used_context: bool,
        response_chars: int,
    ) -> None:
        reasons_blob = "\n".join(quality_reasons or [])
        with self._lock:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO improvement_episodes
                    (intent, quality_score, quality_reasons, was_refined, used_context, response_chars)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent,
                        int(quality_score),
                        reasons_blob,
                        1 if was_refined else 0,
                        1 if used_context else 0,
                        int(response_chars),
                    ),
                )

    def guidance_for_intent(self, intent: str, sample_size: int = 60) -> List[str]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT quality_score, quality_reasons, response_chars
                FROM improvement_episodes
                WHERE intent = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (intent, int(sample_size)),
            ).fetchall()

        if not rows:
            return []

        reason_blob = "\n".join((row["quality_reasons"] or "") for row in rows).lower()
        avg_chars = sum(int(row["response_chars"] or 0) for row in rows) / max(1, len(rows))
        guidance: List[str] = []

        if reason_blob.count("missing structured presentation") >= 3:
            guidance.append("Use clear section headers with concise bullets for dense prompts.")
        if reason_blob.count("missing actionable next steps") >= 2:
            guidance.append("Always close with concrete next steps that can be executed immediately.")
        if reason_blob.count("limited explicit reasoning depth") >= 3:
            guidance.append("Show explicit cause/effect reasoning, assumptions, and trade-offs.")
        if reason_blob.count("too short for an educated response") >= 2 or avg_chars < 350:
            guidance.append("Increase depth with practical details, not filler.")
        if reason_blob.count("did not clearly ground answer in ingested context") >= 2:
            guidance.append("When context exists, explicitly cite what was used from ingested knowledge.")

        return guidance[:4]

    def close(self) -> None:
        with self._lock:
            self.conn.close()


DEFAULT_NEWS_SOURCES = [
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "http://feeds.foxnews.com/foxnews/latest",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.npr.org/1001/rss.xml",
    "https://feeds.skynews.com/feeds/rss/home.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.feedburner.com/TheHackersNews",
    "https://krebsonsecurity.com/feed/",
    "https://www.darkreading.com/rss.xml",
]

SOURCE_SITES_FILE = "knowledge_sources.json"
MAX_FEED_ITEMS_PER_SOURCE = 6
MAX_SITE_PAGES_PER_SOURCE = 10
MAX_CHAT_LEARNING_CHARS = 6000
MAX_KNOWLEDGE_CONTEXT_CHARS = 12_000
MAX_KNOWLEDGE_SNIPPETS_PER_QUERY = 8


class PortableLLM:
    """Portable LLM orchestrator with quality-controlled responses."""

    def __init__(
        self,
        db_path: str = "llm_portable_conversations.db",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_offline_fallback: bool = False,
        strict_local_only: bool = True,
        system_prompt: str = (
            "You are Perseus, a smart, practical technical assistant. "
            "Provide accurate, genuine, context-aware responses with candid, intelligent feedback. "
            "Act as the user's personal knowledge assistant: prefer learned source material, local files, "
            "ingested context, and relevant user-provided chat knowledge over sending the user to browse manually. "
            "Avoid hollow praise, filler, and generic disclaimers; be useful, honest, and actionable."
        ),
    ):
        self.strict_local_only = bool(strict_local_only)
        self._local_provider_order = ["ollama", "fallback"]
        self.manager = ConversationManager(db_path=db_path)
        self.offline = (
            OfflineLLM(use_knowledge_db=True)
            if use_offline_fallback and not self.strict_local_only
            else None
        )
        self.system_prompt = self._compose_system_prompt(
            system_prompt,
            self._load_monday_personality(),
            self._load_ollama_smart_content(),
        )
        self.web_learner = self._create_web_learner()
        self.improvement_store = SelfImprovementStore()

        self.stats = LLMStats()
        self._quality_threshold = 72
        self._max_history_messages = 20

        self.provider = self._resolve_provider(provider)
        self.model = model or self._default_model_for(self.provider)

        self.conversation = self.manager.create_conversation(
            title="Portable LLM Session",
            provider=self.provider,
            model=self.model,
            system_prompt=self.system_prompt,
        )

    @staticmethod
    def _compose_system_prompt(base_prompt: str, personality_prompt: str, smart_prompt: str) -> str:
        """Combine the base assistant contract with local personality and Ollama smart guidance."""
        sections: List[str] = []
        base = (base_prompt or "").strip()
        personality = (personality_prompt or "").strip()
        smart = (smart_prompt or "").strip()

        if base:
            sections.append(base)
        if personality:
            sections.append(f"Personality layer:\n{personality}")
        if smart:
            sections.append(f"Ollama smart-response guidance:\n{smart}")

        return "\n\n".join(sections)

    @staticmethod
    def _load_monday_personality() -> str:
        """Load Monday's personality prompt from the companion text file without executing it."""
        path = Path(__file__).resolve().parent / MONDAY_PERSONALITY_FILE
        if not path.exists():
            return ""

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Unable to read Monday personality file: %s", exc)
            return ""

        system_match = re.search(
            r"def system_prompt\(self\).*?return f\"\"\"(.*?)\"\"\"\.strip\(\)",
            raw,
            flags=re.DOTALL,
        )
        style_match = re.search(
            r"return \"\"\"\s*(Use balanced Monday:.*?occasionally poetic\.)\s*\"\"\"\.strip\(\)",
            raw,
            flags=re.DOTALL,
        )

        if not system_match:
            return raw.strip()

        prompt = system_match.group(1).strip()
        prompt = prompt.replace("{self.name}", "Monday")
        prompt = prompt.replace("{self.role}", "skeptical but loyal technical co-pilot")
        prompt = prompt.replace("{self.tone_mode}", "balanced")

        if style_match:
            prompt = f"{prompt}\n\n{style_match.group(1).strip()}"

        return prompt

    @staticmethod
    def _load_ollama_smart_content() -> str:
        """Load the smart Ollama response contract from the companion text file without executing it."""
        path = Path(__file__).resolve().parent / OLLAMA_SMART_CONTENT_FILE
        if not path.exists():
            return ""

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Unable to read Ollama smart content file: %s", exc)
            return ""

        try:
            tree = ast.parse(raw)
        except SyntaxError as exc:
            logger.warning("Unable to parse Ollama smart content file: %s", exc)
            return raw.strip()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
                continue
            if not node.args or not isinstance(node.args[0], ast.Dict):
                continue

            values = {}
            for key, value in zip(node.args[0].keys, node.args[0].values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    try:
                        values[key.value] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        continue

            if values.get("role") == "system" and isinstance(values.get("content"), str):
                return values["content"].strip()

        return raw.strip()

    def available_providers(self) -> List[str]:
        """Return available providers from existing conversation core."""
        return self.manager.get_available_providers()

    def ask(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Generate a high-quality response with refinement and failover."""
        response, _meta = self.ask_with_metadata(prompt, temperature=temperature, max_tokens=max_tokens)
        return response

    def ask_with_metadata(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, object]]:
        """Generate a response and return diagnostics metadata."""
        if not prompt or not prompt.strip():
            return "Please provide a prompt.", {"error": "empty_prompt"}

        prompt = prompt.strip()
        self.stats.total_requests += 1

        if temperature is not None:
            self.conversation.temperature = float(temperature)
        if max_tokens is not None:
            self.conversation.max_tokens = int(max_tokens)

        profile = self._profile_prompt(prompt)
        enriched = self._enrich_prompt_with_knowledge(prompt)
        self.conversation.add_message(
            "user",
            prompt,
            metadata={"intent": profile.intent, "complexity": profile.complexity},
        )
        self.manager._save_conversation(self.conversation)

        response, provider_used, quality, refined = self._generate_best_response(
            enriched.text,
            profile,
            has_context=enriched.has_context,
        )

        used_offline = False
        if not response and self.offline:
            used_offline = True
            self.stats.offline_fallbacks += 1
            logger.warning("No provider response available; using OfflineLLM fallback")
            response = self.offline.generate(user_input=prompt, mood=profile.mood, system_prompt=self.system_prompt)
            provider_used = "offline"
            quality = self._assess_quality(response, profile)

        if enriched.has_context and response and quality.score < self._quality_threshold:
            response = self._build_grounded_response(prompt=prompt, context=enriched.context_preview)
            provider_used = "grounded-fallback"
            quality = self._assess_quality(response, profile, has_context=True)
            refined = True

        if response and str(response).strip():
            model_used = self.model if provider_used == self.provider else self._default_model_for(provider_used)
            self.improvement_store.record(
                intent=profile.intent,
                quality_score=quality.score,
                quality_reasons=quality.reasons,
                was_refined=refined,
                used_context=enriched.has_context,
                response_chars=len(str(response)),
            )
            self.conversation.add_message(
                "assistant",
                response,
                metadata={
                    "provider": provider_used,
                    "model": model_used,
                    "quality_score": quality.score,
                    "quality_reasons": quality.reasons,
                    "refined": refined,
                    "offline": used_offline,
                    "grounded_with_ingested_context": enriched.has_context,
                },
            )
            self.manager._save_conversation(self.conversation)
            self._update_quality_average(quality.score)

            self._learn_from_chat_turn(
                prompt=prompt,
                response=str(response),
                profile=profile,
                provider=provider_used,
                quality=quality,
            )

            metadata = {
                "provider": provider_used,
                "model": model_used,
                "quality_score": quality.score,
                "quality_reasons": quality.reasons,
                "refined": refined,
                "offline_fallback": used_offline,
                "intent": profile.intent,
                "complexity": profile.complexity,
                "grounded_with_ingested_context": enriched.has_context,
                "context_preview": enriched.context_preview,
                "strict_local_only": self.strict_local_only,
            }
            return response, metadata

        self.stats.provider_failures += 1
        return "No response generated by the active provider.", {
            "provider": "none",
            "quality_score": 0,
            "quality_reasons": ["All providers failed"],
        }

    @staticmethod
    def _build_grounded_response(prompt: str, context: str) -> str:
        """Deterministic rescue response when grounded output quality is too low."""
        return (
            "Summary:\n"
            "Based on currently ingested reporting, there is relevant context tied to your request. "
            "The key takeaway is to treat this as actionable but still validate details against primary sources before operational changes.\n\n"
            "Ingested Context Used:\n"
            f"- {context or 'Relevant ingested context was retrieved from configured news/security sources.'}\n\n"
            "Educational Notes:\n"
            "- Why this matters: timely ingestion helps detect trends and incidents earlier.\n"
            "- Practical implication: prioritize verification, patch cadence, and communications based on severity.\n"
            "- Uncertainty: news summaries can omit technical depth; confirm with vendor advisories/CVE records.\n\n"
            "Next Steps:\n"
            "1. Cross-check the claim with primary technical references.\n"
            "2. Map impact to your environment.\n"
            "3. Execute mitigations and monitor for updates.\n\n"
            f"Original question: {prompt}"
        )

    def get_stats(self) -> Dict[str, float]:
        """Return runtime health metrics for this session."""
        return {
            "total_requests": self.stats.total_requests,
            "provider_successes": self.stats.provider_successes,
            "provider_failures": self.stats.provider_failures,
            "refinements_used": self.stats.refinements_used,
            "offline_fallbacks": self.stats.offline_fallbacks,
            "average_quality": round(self.stats.average_quality, 2),
        }

    def set_provider(self, provider: str, model: Optional[str] = None) -> bool:
        """Switch provider for the active conversation."""
        if self.strict_local_only and provider not in self._local_provider_order:
            return False
        selected_model = model or self._default_model_for(provider)
        switched = self.manager.switch_provider(self.conversation.id, provider, selected_model)
        if switched:
            self.provider = provider
            self.model = selected_model
            self.conversation.provider = provider
            self.conversation.model = selected_model
        return switched

    def close(self) -> None:
        """Release local resources."""
        self.improvement_store.close()
        if self.web_learner:
            self.web_learner.close()
        if self.offline:
            self.offline.close()

    def ingest_web_content(self, url: str, content: str, title: str = "") -> Dict[str, object]:
        """Ingest webpage content into knowledge store for future conversations."""
        if not self.web_learner:
            return {"ok": False, "error": "WebKnowledgeLearner not available"}

        try:
            result = self.web_learner.learn_from_content(url=url, content=content, metadata={"title": title})
            return {"ok": True, "summary": result}
        except Exception as exc:
            # Keep learner enabled on single ingest errors so later sources can still be learned.
            logger.warning("Web content ingest failed for %s: %s", url, exc)
            return {"ok": False, "error": str(exc)}

    def _learn_from_chat_turn(
        self,
        prompt: str,
        response: str,
        profile: PromptProfile,
        provider: str,
        quality: ResponseQuality,
    ) -> None:
        """Persist useful chat turns so future answers adapt to the user's knowledge and preferences."""
        if not self.web_learner:
            return

        prompt_text = (prompt or "").strip()
        response_text = (response or "").strip()
        if not prompt_text:
            return

        low_value_responses = [
            "i do not have enough learned context",
            "no response generated",
        ]
        if any(marker in response_text.lower() for marker in low_value_responses):
            response_text = ""

        content = (
            "Learned chat interaction:\n"
            f"Timestamp: {datetime.utcnow().isoformat(timespec='seconds')}Z\n"
            f"Intent: {profile.intent}\n"
            f"Complexity: {profile.complexity}\n"
            f"Provider: {provider}\n"
            f"Quality score: {quality.score}\n\n"
            "User input:\n"
            f"{prompt_text}\n"
        )
        if response_text:
            content += "\nAssistant response:\n" + response_text

        content = content[:MAX_CHAT_LEARNING_CHARS]
        title = f"chat-memory/{profile.intent}/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", prompt_text[:80]).strip("-") or "interaction"
        url = f"perseus://chat-memory/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}/{safe_id}"

        try:
            self.ingest_web_content(url=url, content=content, title=title)
        except Exception as exc:
            logger.warning("Chat learning failed: %s", exc)

    def ingest_url(self, url: str, timeout: int = 15) -> Dict[str, object]:
        """Fetch a URL and ingest extracted text into the learning store."""
        try:
            page = fetch_url_text(url, timeout=timeout)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url}

        ingest = self.ingest_web_content(url=url, content=page["text"], title=page.get("title", ""))
        ingest["url"] = url
        ingest["title"] = page.get("title", "")
        return ingest

    def ingest_folder(
        self,
        folder_path: str = DEFAULT_KNOWLEDGE_FOLDER,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        max_file_bytes: int = MAX_KNOWLEDGE_FILE_BYTES,
    ) -> Dict[str, object]:
        """Ingest supported text files from a local folder into the knowledge store."""
        if not self.web_learner:
            return {"ok": False, "error": "WebKnowledgeLearner not available", "folder": folder_path}

        root = Path(folder_path).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parent / root
        if not root.exists():
            return {"ok": False, "error": "Folder does not exist", "folder": str(root)}
        if not root.is_dir():
            return {"ok": False, "error": "Path is not a folder", "folder": str(root)}

        allowed = {
            "" if not ext else ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (extensions or list(SUPPORTED_KNOWLEDGE_EXTENSIONS))
        }
        files = sorted(root.rglob("*") if recursive else root.glob("*"))

        results: List[Dict[str, object]] = []
        learned_titles: List[str] = []
        successes = 0
        skipped = 0

        for path in files:
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed:
                skipped += 1
                continue

            try:
                size = path.stat().st_size
            except OSError as exc:
                skipped += 1
                results.append({"ok": False, "path": str(path), "error": str(exc)})
                continue

            if size <= 0:
                skipped += 1
                results.append({"ok": False, "path": str(path), "error": "Empty file"})
                continue
            if size > max_file_bytes:
                skipped += 1
                results.append({"ok": False, "path": str(path), "error": f"File exceeds {max_file_bytes} bytes"})
                continue

            try:
                content = _read_knowledge_file(path)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                skipped += 1
                results.append({"ok": False, "path": str(path), "error": str(exc)})
                continue

            if not content.strip():
                skipped += 1
                results.append({"ok": False, "path": str(path), "error": "No text content"})
                continue

            try:
                relative_title = str(path.relative_to(root))
            except ValueError:
                relative_title = path.name

            title = f"{root.name}/{relative_title}"

            ingest = self.ingest_web_content(url=path.resolve().as_uri(), content=content, title=title)
            ingest["path"] = str(path)
            ingest["title"] = title
            results.append(ingest)
            if ingest.get("ok"):
                successes += 1
                learned_titles.append(title)

        if successes > 0:
            index_content = _build_folder_index_content(root=root, files=files, learned_titles=learned_titles)
            index_ingest = self.ingest_web_content(
                url=f"perseus://folder-index/{root.name}",
                content=index_content,
                title=f"{root.name}/_folder_index",
            )
            index_ingest["path"] = str(root)
            index_ingest["title"] = f"{root.name}/_folder_index"
            results.append(index_ingest)

        return {
            "ok": successes > 0,
            "folder": str(root),
            "recursive": recursive,
            "total_files_seen": len([path for path in files if path.is_file()]),
            "successes": successes,
            "failures": len([item for item in results if not item.get("ok")]),
            "skipped": skipped,
            "extensions": sorted(allowed),
            "results": results,
        }

    @staticmethod
    def _source_sites_path() -> Path:
        return Path(__file__).resolve().parent / SOURCE_SITES_FILE

    def load_source_sites(self) -> List[str]:
        """Load the persisted personal source-site list."""
        path = self._source_sites_path()
        if not path.exists():
            return list(DEFAULT_NEWS_SOURCES)

        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Unable to read source sites: %s", exc)
            return list(DEFAULT_NEWS_SOURCES)

        if not raw:
            return list(DEFAULT_NEWS_SOURCES)

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                items = parsed.get("sources", [])
            else:
                items = parsed
            sources = [str(item).strip() for item in items if str(item).strip()]
            return sources or list(DEFAULT_NEWS_SOURCES)
        except json.JSONDecodeError:
            sources = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
            return sources or list(DEFAULT_NEWS_SOURCES)

    def save_source_sites(self, sources: List[str]) -> Dict[str, object]:
        """Persist the personal source-site list used by the ingest tab."""
        cleaned: List[str] = []
        seen = set()
        for source in sources:
            item = str(source).strip()
            if not item or item.startswith("#"):
                continue
            if not urlparse(item).scheme:
                item = f"https://{item}"
            if item in seen:
                continue
            cleaned.append(item)
            seen.add(item)

        path = self._source_sites_path()
        payload = {
            "sources": cleaned,
            "notes": "Feeds and normal websites are supported. Normal websites are ingested with a bounded same-site link scan.",
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc), "path": str(path)}

        return {"ok": True, "path": str(path), "sources": cleaned, "count": len(cleaned)}

    def ingest_source_sites(
        self,
        sources: Optional[List[str]] = None,
        timeout: int = 15,
        max_pages_per_source: int = MAX_SITE_PAGES_PER_SOURCE,
    ) -> Dict[str, object]:
        """Ingest configured feeds and websites as personal assistant source material."""
        source_list = sources or self.load_source_sites()
        results: List[Dict[str, object]] = []
        successes = 0
        total_entries_ingested = 0
        total_entry_failures = 0

        for source in source_list:
            result = self.ingest_source_site(source, timeout=timeout, max_pages=max_pages_per_source)
            results.append(result)
            if result.get("ok"):
                successes += 1
                total_entries_ingested += int(result.get("entry_successes", 0))
                total_entry_failures += int(result.get("entry_failures", 0))

        return {
            "total": len(source_list),
            "successes": successes,
            "failures": len(source_list) - successes,
            "entry_successes": total_entries_ingested,
            "entry_failures": total_entry_failures,
            "results": results,
        }

    def ingest_news_sources(self, sources: Optional[List[str]] = None, timeout: int = 15) -> Dict[str, object]:
        """Backward-compatible alias for source-site ingestion."""
        return self.ingest_source_sites(sources=sources, timeout=timeout)

    def ingest_feed_source(self, source_url: str, timeout: int = 15) -> Dict[str, object]:
        """Backward-compatible alias for one source URL."""
        return self.ingest_source_site(source_url, timeout=timeout)

    def ingest_source_site(
        self,
        source_url: str,
        timeout: int = 15,
        max_pages: int = MAX_SITE_PAGES_PER_SOURCE,
    ) -> Dict[str, object]:
        """Ingest a feed or normal website plus a bounded set of linked same-site pages."""
        if not urlparse(source_url).scheme:
            source_url = f"https://{source_url}"

        try:
            fetched = fetch_url_payload(source_url, timeout=timeout)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": source_url}

        feed_ingest = self.ingest_web_content(
            url=source_url,
            content=fetched["text"],
            title=fetched.get("title", ""),
        )
        if not feed_ingest.get("ok"):
            feed_ingest["url"] = source_url
            feed_ingest["title"] = fetched.get("title", "")
            return feed_ingest

        links = _extract_feed_links(fetched.get("raw", ""), source_url)
        source_type = "feed" if links else "site"
        if links:
            links = links[:MAX_FEED_ITEMS_PER_SOURCE]
        else:
            links = _extract_site_links(fetched.get("raw", ""), source_url)[:max_pages]

        entry_results: List[Dict[str, object]] = []
        entry_successes = 0

        for link in links:
            result = self.ingest_url(link, timeout=timeout)
            entry_results.append(result)
            if result.get("ok"):
                entry_successes += 1

        return {
            "ok": True,
            "url": source_url,
            "title": fetched.get("title", ""),
            "source_type": source_type,
            "summary": feed_ingest.get("summary", {}),
            "entries_discovered": len(links),
            "entry_successes": entry_successes,
            "entry_failures": len(links) - entry_successes,
            "entry_results": entry_results,
        }

    def _create_web_learner(self):
        if not WebKnowledgeLearner:
            return None
        try:
            learner = WebKnowledgeLearner(db_path=WEB_LEARNING_DB_PATH)
            stats = learner.store.get_learning_stats()
            if not isinstance(stats, dict):
                raise RuntimeError("Failed to initialize learning stats")
            return learner
        except Exception as exc:
            logger.warning("WebKnowledgeLearner unavailable: %s", exc)
            return None

    def _disable_web_learner(self) -> None:
        if not self.web_learner:
            return
        try:
            self.web_learner.close()
        except Exception:
            pass
        self.web_learner = None

    def _enrich_prompt_with_knowledge(self, prompt: str) -> EnrichedPrompt:
        """Inject relevant learned web context into the prompt when available."""
        if _is_small_talk_prompt(prompt) or _is_capability_prompt(prompt) or _is_general_knowledge_prompt(prompt):
            return EnrichedPrompt(text=prompt, has_context=False)

        if not self.web_learner:
            return EnrichedPrompt(text=prompt, has_context=False)

        try:
            context = self._lookup_knowledge_context(prompt)
        except Exception as exc:
            logger.warning("Knowledge context lookup failed: %s", exc)
            return EnrichedPrompt(text=prompt, has_context=False)

        if not context:
            return EnrichedPrompt(text=prompt, has_context=False)

        preview = context[:260].replace("\n", " ").strip()
        enriched_text = (
            "You have retrieved ingested web knowledge relevant to this query. "
            "Ground your answer in that ingested context, including learned user chat memory when relevant, "
            "and be explicit when you do.\n\n"
            "Ingested context:\n"
            f"{context}\n\n"
            "Output requirements:\n"
            "1. Start with a realistic summary of what is currently known.\n"
            "2. Review all retrieved ingested context blocks, synthesize every relevant non-duplicate point, "
            "and include an 'Ingested Context Used' section with concrete points.\n"
            "3. Include uncertainty where evidence is incomplete.\n"
            "4. Use learned user preferences, prior chat facts, and project context when they are relevant.\n"
            "5. Provide educational explanation and practical implications.\n"
            "6. End with a 'Next Steps' section focused on action.\n\n"
            "User request:\n"
            f"{prompt}"
        )

        return EnrichedPrompt(text=enriched_text, has_context=True, context_preview=preview)

    def _lookup_knowledge_context(self, prompt: str) -> str:
        """Try multiple targeted queries and retain as much relevant learned content as the context budget allows."""
        queries = self._knowledge_queries(prompt)
        collected: List[str] = []
        seen = set()
        remaining_chars = MAX_KNOWLEDGE_CONTEXT_CHARS
        broad_request = self._requests_broad_knowledge(prompt)

        if broad_request:
            for snippet in self._split_knowledge_context(self._lookup_all_knowledge_context()):
                if not snippet or snippet in seen:
                    continue

                budgeted = snippet[:remaining_chars].strip()
                if not budgeted:
                    continue

                collected.append(budgeted)
                seen.add(snippet)
                remaining_chars -= len(budgeted) + 2
                if remaining_chars <= 0:
                    break

        for q in queries:
            if not q or remaining_chars <= 0:
                continue

            try:
                context = self.web_learner.get_knowledge_context_for_query(q, limit=MAX_KNOWLEDGE_SNIPPETS_PER_QUERY)
            except TypeError:
                context = self.web_learner.get_knowledge_context_for_query(q)

            for snippet in self._split_knowledge_context(context):
                if not snippet or snippet in seen:
                    continue

                budgeted = snippet[:remaining_chars].strip()
                if not budgeted:
                    continue

                collected.append(budgeted)
                seen.add(snippet)
                remaining_chars -= len(budgeted) + 2
                if remaining_chars <= 0:
                    break

            if remaining_chars <= 0:
                break

        if remaining_chars > 0 and not collected:
            for snippet in self._split_knowledge_context(self._lookup_all_knowledge_context()):
                if not snippet or snippet in seen:
                    continue

                budgeted = snippet[:remaining_chars].strip()
                if not budgeted:
                    continue

                collected.append(budgeted)
                seen.add(snippet)
                remaining_chars -= len(budgeted) + 2
                if remaining_chars <= 0:
                    break

        return "\n\n".join(collected).strip()

    def _lookup_all_knowledge_context(self) -> str:
        """Retrieve recent learned source blocks when the prompt calls for broad ingested knowledge."""
        getter = getattr(self.web_learner, "get_all_knowledge_context", None)
        if callable(getter):
            try:
                return getter(limit=MAX_KNOWLEDGE_SNIPPETS_PER_QUERY)
            except TypeError:
                return getter()

        conn = getattr(self.web_learner, "conn", None)
        if not conn:
            return ""

        try:
            rows = conn.execute(
                """
                SELECT title, content
                FROM learned_documents
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (MAX_KNOWLEDGE_SNIPPETS_PER_QUERY,),
            ).fetchall()
        except Exception:
            return ""

        snippets: List[str] = []
        for row in rows:
            title = row["title"] if isinstance(row, sqlite3.Row) else row[0]
            content = row["content"] if isinstance(row, sqlite3.Row) else row[1]
            normalized = re.sub(r"\s+", " ", str(content or "")).strip()[:900]
            snippets.append(f"Source: {title}\n{normalized}")

        return "\n\n".join(snippets)

    @staticmethod
    def _requests_broad_knowledge(prompt: str) -> bool:
        lower = prompt.lower()
        broad_markers = [
            "all ingested",
            "all learned",
            "everything ingested",
            "everything learned",
            "knowledge base",
            "learned content",
            "ingested content",
        ]
        return any(marker in lower for marker in broad_markers)

    @staticmethod
    def _split_knowledge_context(context: str) -> List[str]:
        """Split returned learner context into source-sized blocks for dedupe and budget handling."""
        text = (context or "").strip()
        if not text:
            return []
        blocks = [block.strip() for block in re.split(r"\n\s*\n(?=Source: )", text) if block.strip()]
        return blocks or [text]

    @staticmethod
    def _knowledge_queries(prompt: str) -> List[str]:
        """Extract likely lookup terms from prompt (CVE/CWE + key tokens)."""
        lower = prompt.lower()
        queries: List[str] = []

        cve_matches = re.findall(r"cve-\d{4}-\d{4,}", lower, flags=re.IGNORECASE)
        cwe_matches = re.findall(r"cwe-\d+", lower, flags=re.IGNORECASE)
        for m in cve_matches + cwe_matches:
            queries.append(m.upper())

        tokens = re.findall(r"[a-zA-Z]{4,}", lower)
        stop = {"what", "about", "that", "with", "from", "this", "have", "does", "into", "their", "they", "them", "explain", "summarize"}
        keywords = [t for t in tokens if t not in stop]
        queries.extend(keywords[:4])

        if len(keywords) >= 2:
            queries.append(f"{keywords[0]} {keywords[1]}")

        queries.append(prompt)
        return queries

    def _profile_prompt(self, prompt: str) -> PromptProfile:
        """Classify user prompt to tailor prompting strategy."""
        lower = prompt.lower()
        words = prompt.split()

        if any(token in lower for token in ["feedback", "critique", "review", "thoughts", "opinion", "advise", "advice"]):
            intent = "feedback"
        elif any(token in lower for token in ["compare", "tradeoff", "trade-off", "evaluate", "analyze", "analysis"]):
            intent = "analytical"
        elif any(token in lower for token in ["plan", "strategy", "roadmap", "prioritize", "recommend"]):
            intent = "strategic"
        elif any(token in lower for token in ["teach", "explain", "what is", "how does", "why"]):
            intent = "educational"
        else:
            intent = "technical"

        complexity = "high" if len(words) > 20 or any(t in lower for t in ["architecture", "constraints", "production", "distributed"]) else "normal"
        mood = "analytical" if intent in {"technical", "analytical", "feedback"} else "pragmatic"
        conversational_markers = ["chat", "talk", "casual", "normal", "plain english", "simple terms"]
        concise_markers = ["brief", "short", "concise", "tldr", "quick answer", "one-liner", "one line"]
        structured_markers = ["steps", "plan", "outline", "table", "bullet", "checklist", "roadmap", "compare"]

        prefer_concise = any(marker in lower for marker in concise_markers)
        prefer_structure = any(marker in lower for marker in structured_markers) or intent in {"strategic", "analytical", "feedback"}
        conversational = any(marker in lower for marker in conversational_markers) or not prefer_structure

        if intent == "feedback":
            expected_shape = "feedback"
        elif prefer_structure:
            expected_shape = "structured"
        elif prefer_concise:
            expected_shape = "concise"
        elif conversational:
            expected_shape = "conversational"
        else:
            expected_shape = "didactic"

        return PromptProfile(
            intent=intent,
            complexity=complexity,
            mood=mood,
            expected_shape=expected_shape,
            prefer_structure=prefer_structure,
            prefer_concise=prefer_concise,
            conversational=conversational,
        )

    def _provider_candidates(self) -> List[str]:
        """Rank providers for this request, preserving selected primary first."""
        ordered = [self.provider]
        if self.strict_local_only:
            ordered.extend(self._local_provider_order)
        else:
            ordered.extend(["ollama", "openai", "mistral", "azure", "fallback"])
        seen = set()
        candidates: List[str] = []

        for name in ordered:
            if name in seen:
                continue
            prov = self.manager.providers.get(name)
            if prov and prov.available:
                candidates.append(name)
                seen.add(name)

        return candidates

    def _generate_best_response(
        self,
        prompt: str,
        profile: PromptProfile,
        has_context: bool = False,
    ) -> Tuple[Optional[str], str, ResponseQuality, bool]:
        """Generate response with quality checks, refinement, and provider failover."""
        best_response: Optional[str] = None
        best_provider = "none"
        best_quality = ResponseQuality(score=0, reasons=["No response generated"])
        was_refined = False

        for provider_name in self._provider_candidates():
            raw = self._generate_with_provider(provider_name, prompt, profile, refine=False, prior_response=None)
            quality = self._assess_quality(raw, profile, has_context=has_context)

            if raw and quality.score >= self._quality_threshold:
                self.stats.provider_successes += 1
                if provider_name != self.provider:
                    self.provider = provider_name
                    self.model = self._default_model_for(provider_name)
                return raw.strip(), provider_name, quality, False

            if raw and quality.score > best_quality.score:
                best_response, best_provider, best_quality = raw.strip(), provider_name, quality

            if raw and provider_name != "fallback" and quality.score < self._quality_threshold:
                refined = self._generate_with_provider(provider_name, prompt, profile, refine=True, prior_response=raw)
                refined_quality = self._assess_quality(refined, profile, has_context=has_context)

                if refined_quality.score > quality.score:
                    self.stats.refinements_used += 1
                    raw = refined
                    quality = refined_quality
                    was_refined = True

                if raw and quality.score >= self._quality_threshold:
                    self.stats.provider_successes += 1
                    if provider_name != self.provider:
                        self.provider = provider_name
                        self.model = self._default_model_for(provider_name)
                    return raw.strip(), provider_name, quality, was_refined

                if raw and quality.score > best_quality.score:
                    best_response, best_provider, best_quality = raw.strip(), provider_name, quality

        if best_response:
            self.stats.provider_successes += 1
            return best_response, best_provider, best_quality, was_refined

        self.stats.provider_failures += 1
        return None, "none", best_quality, was_refined

    def _generate_with_provider(
        self,
        provider_name: str,
        prompt: str,
        profile: PromptProfile,
        refine: bool,
        prior_response: Optional[str],
    ) -> Optional[str]:
        """Generate content from a provider using adaptive prompt contracts."""
        provider = self.manager.providers.get(provider_name)
        if not provider or not provider.available:
            return None

        messages = self._build_messages(prompt=prompt, profile=profile, refine=refine, prior_response=prior_response)
        model = self.model if provider_name == self.provider else self._default_model_for(provider_name)

        try:
            return provider.generate(
                prompt,
                messages=messages,
                model=model,
                temperature=self.conversation.temperature,
                max_tokens=self.conversation.max_tokens,
            )
        except Exception as exc:
            logger.error("Provider %s generation error: %s", provider_name, exc)
            return None

    def _build_messages(
        self,
        prompt: str,
        profile: PromptProfile,
        refine: bool,
        prior_response: Optional[str],
    ) -> List[Dict[str, str]]:
        """Create API-formatted messages with adaptive instructions and short memory."""
        response_contract = [
            "Respond like a capable general-purpose assistant: natural, thoughtful, clear, and directly useful.",
            "Match the user's tone and requested depth.",
            "Give genuine feedback: identify what is strong, what is weak, why it matters, and what to do next.",
            "Do not flatter, over-agree, or pad the answer; be candid without being harsh.",
            "If uncertain, state assumptions, confidence, and what data is needed instead of inventing details.",
        ]

        if profile.intent == "feedback":
            response_contract.append(
                "For feedback requests, lead with the core judgment, then give specific evidence, trade-offs, and prioritized improvements."
            )

        if profile.prefer_structure:
            response_contract.append(
                "Use structured formatting (sections or bullets) when it improves clarity, including concrete next steps."
            )
        elif profile.prefer_concise:
            response_contract.append("Keep the answer concise while still being specific and accurate.")
        else:
            response_contract.append(
                "Default to natural prose; add light structure only when it helps readability."
            )

        response_contract.append(
            "If ingested web context is present in the prompt, include a short 'Ingested Context Used' section "
            "with concrete facts from that context."
        )

        system = (
            f"{self.system_prompt}\n"
            f"Mode: {profile.intent}. Complexity target: {profile.complexity}. Output shape: {profile.expected_shape}.\n"
            f"{' '.join(response_contract)}"
        )

        learned_guidance = self.improvement_store.guidance_for_intent(profile.intent)
        if learned_guidance:
            guidance_block = "\n".join(f"- {item}" for item in learned_guidance)
            system += f"\nSelf-improvement directives from prior sessions:\n{guidance_block}"

        if refine and prior_response:
            system += (
                "\nImprove the previous draft by increasing specificity, correctness, practical detail, and genuine judgment. "
                "Remove generic phrasing, hollow encouragement, and unsupported claims; tighten structure."
            )

        history = self.conversation.messages[-self._max_history_messages :]
        api_messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        api_messages.extend({"role": msg.role, "content": msg.content} for msg in history)

        if refine and prior_response:
            api_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Refine this draft response while preserving factual accuracy and making it more insightful:\n\n"
                        f"{prior_response}"
                    ),
                }
            )

        return api_messages

    def _assess_quality(
        self,
        response: Optional[str],
        profile: PromptProfile,
        has_context: bool = False,
    ) -> ResponseQuality:
        """Score response quality using lightweight heuristics."""
        if not response or not str(response).strip():
            return ResponseQuality(score=0, reasons=["Empty response"])

        text = str(response).strip()
        lower = text.lower()
        score = 100
        reasons: List[str] = []

        generic_markers = [
            "i don't have specific knowledge",
            "as an ai language model",
            "cannot provide",
            "i'm unable to",
            "i do not have access",
        ]
        filler_markers = [
            "great question",
            "that's a great question",
            "you're absolutely right",
            "it depends" if len(text) < 220 else "",
        ]

        min_len = 60 if profile.prefer_concise else 90
        brief_len = 110 if profile.prefer_concise else 150

        if len(text) < min_len:
            score -= 30
            reasons.append("Too short for an educated response")
        elif len(text) < brief_len:
            score -= 12
            reasons.append("Response is brief for sophisticated depth")

        if any(marker in lower for marker in generic_markers):
            score -= 35
            reasons.append("Contains generic low-value fallback phrasing")

        if any(marker and marker in lower for marker in filler_markers):
            score -= 10
            reasons.append("Contains filler or hollow agreement instead of direct value")

        if profile.prefer_structure or profile.intent in {"strategic", "analytical"}:
            has_structure = any(token in text for token in ["\n-", "\n1.", ":\n", "##"])
            if not has_structure:
                score -= 8
                reasons.append("Missing structured presentation")

            if "next steps" not in lower:
                score -= 6
                reasons.append("Missing actionable next steps")

            reasoning_signals = ["because", "therefore", "trade-off", "risk", "assumption"]
            if not any(marker in lower for marker in reasoning_signals):
                score -= 8
                reasons.append("Limited explicit reasoning depth")
        elif profile.intent == "technical":
            technical_signals = ["for example", "because", "in practice", "trade-off", "risk"]
            if not any(marker in lower for marker in technical_signals):
                score -= 4
                reasons.append("Could use slightly more practical reasoning")

        if profile.intent == "feedback":
            feedback_signals = [
                "because",
                "risk",
                "trade-off",
                "recommend",
                "improve",
                "strong",
                "weak",
                "next",
            ]
            if sum(1 for marker in feedback_signals if marker in lower) < 3:
                score -= 14
                reasons.append("Feedback is not specific or actionable enough")

            judgment_signals = ["i would", "my read", "the issue", "the strongest", "the weakest", "priority"]
            if not any(marker in lower for marker in judgment_signals):
                score -= 8
                reasons.append("Missing a clear, candid judgment")

        if "?" in text and len(text) < 140:
            score -= 8
            reasons.append("Likely incomplete answer")

        if has_context:
            grounded_markers = [
                "ingested context used",
                "based on ingested",
                "from ingested",
                "from the ingested",
                "according to ingested",
            ]
            if not any(marker in lower for marker in grounded_markers):
                score -= 15
                reasons.append("Did not clearly ground answer in ingested context")

            educational_markers = ["summary", "implication", "why", "because", "in practice", "next steps"]
            if sum(1 for marker in educational_markers if marker in lower) < 2:
                score -= 10
                reasons.append("Insufficient educational framing for grounded response")

        if score >= 85:
            reasons.append("Strong depth and structure")
        elif score >= self._quality_threshold:
            reasons.append("Acceptable quality")

        score = max(0, min(100, score))
        return ResponseQuality(score=score, reasons=reasons)

    def _update_quality_average(self, quality_score: int) -> None:
        """Keep a rolling average quality score for this session."""
        n = self.stats.total_requests
        if n <= 0:
            return
        prev = self.stats.average_quality
        self.stats.average_quality = ((prev * (n - 1)) + quality_score) / n

    def _pick_provider(self) -> str:
        """Pick the best available provider with portability in mind."""
        available = self.manager.get_available_providers()
        if self.strict_local_only:
            for candidate in self._local_provider_order:
                if candidate in available:
                    return candidate
            raise RuntimeError(
                "Strict local-only mode is enabled, but no local provider is available. "
                "Install/start Ollama or disable strict mode."
            )

        preferred_order = ["ollama", "openai", "mistral", "azure", "fallback"]
        for candidate in preferred_order:
            if candidate in available:
                return candidate
        return "fallback"

    def _resolve_provider(self, provider: Optional[str]) -> str:
        if provider:
            if self.strict_local_only and provider not in self._local_provider_order:
                raise ValueError(
                    f"Provider '{provider}' is not allowed in strict local-only mode. "
                    f"Allowed providers: {', '.join(self._local_provider_order)}"
                )
            return provider
        return self._pick_provider()

    @staticmethod
    def _default_model_for(provider: str) -> str:
        model_map = {
            "openai": "gpt-3.5-turbo",
            "mistral": "mistral-tiny",
            "ollama": "llama3.2",
            "azure": "gpt-35-turbo",
            "fallback": "fallback",
            "offline": "offline",
        }
        return model_map.get(provider, "fallback")


def launch_portable_llm_chat(
    db_path: str = "llm_portable_conversations.db",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    strict_local_only: bool = True,
    use_offline_fallback: bool = False,
    knowledge_folders: Optional[List[str]] = None,
    auto_ingest_folders: bool = True,
) -> None:
    """Launch a desktop chat window for PortableLLM (no terminal loop)."""
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import ttk
    import webbrowser

    llm = PortableLLM(
        db_path=db_path,
        provider=provider,
        model=model,
        strict_local_only=strict_local_only,
        use_offline_fallback=use_offline_fallback,
    )
    startup_knowledge_folders = knowledge_folders or list(DEFAULT_AUTO_KNOWLEDGE_FOLDERS)
    result_queue: queue.Queue[Tuple[str, Dict[str, object]]] = queue.Queue()

    root = tk.Tk()
    root.title("Perseus Chat")
    root.geometry("1080x760")

    header = ttk.Frame(root, padding=(10, 10, 10, 0))
    header.pack(fill=tk.X)

    title_label = ttk.Label(header, text="Perseus Chat", font=("Segoe UI", 14, "bold"))
    title_label.pack(side=tk.LEFT)

    provider_label = ttk.Label(header, text=f"Provider: {llm.provider} | Model: {llm.model}")
    provider_label.pack(side=tk.RIGHT)

    def show_donation_popup() -> None:
        popup = tk.Toplevel(root)
        popup.title("Support Perseus")
        popup.geometry("460x210")
        popup.transient(root)
        popup.grab_set()

        frame = ttk.Frame(popup, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)

        heading = ttk.Label(frame, text="Support Perseus Development", font=("Segoe UI", 12, "bold"))
        heading.pack(anchor=tk.W)

        body = ttk.Label(
            frame,
            text=(
                "If Perseus is useful to you, donations help keep development and maintenance going.\n\n"
                "Donation link:\nhttps://buy.stripe.com/28EbJ1f7ceo3ckyeES5kk00"
            ),
            justify=tk.LEFT,
        )
        body.pack(anchor=tk.W, pady=(10, 12))

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X)

        def open_donate() -> None:
            webbrowser.open("https://buy.stripe.com/28EbJ1f7ceo3ckyeES5kk00", new=2)

        donate_btn = ttk.Button(actions, text="Donate", command=open_donate)
        donate_btn.pack(side=tk.LEFT)

        close_btn = ttk.Button(actions, text="Maybe Later", command=popup.destroy)
        close_btn.pack(side=tk.LEFT, padx=(8, 0))

    tabs = ttk.Notebook(root)
    tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 8))

    chat_tab = ttk.Frame(tabs)
    ingest_tab = ttk.Frame(tabs)
    tabs.add(chat_tab, text="Chat")
    tabs.add(ingest_tab, text="Knowledge Ingest")

    transcript = tk.Text(chat_tab, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
    transcript.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 8))

    controls = ttk.Frame(chat_tab)
    controls.pack(fill=tk.X)

    input_var = tk.StringVar()
    input_box = ttk.Entry(controls, textvariable=input_var)
    input_box.pack(side=tk.LEFT, fill=tk.X, expand=True)

    status_var = tk.StringVar(value="Ready")
    status_label = ttk.Label(chat_tab, textvariable=status_var, padding=(0, 8, 0, 0))
    status_label.pack(fill=tk.X)

    ingest_top = ttk.Frame(ingest_tab)
    ingest_top.pack(fill=tk.X, pady=(0, 8))

    auto_ingest_var = tk.BooleanVar(value=True)
    auto_ingest_check = ttk.Checkbutton(
        ingest_top,
        text="Auto ingest source sites on start",
        variable=auto_ingest_var,
    )
    auto_ingest_check.pack(side=tk.LEFT)

    ingest_status_var = tk.StringVar(value="Ingest ready")
    ingest_status = ttk.Label(ingest_tab, textvariable=ingest_status_var)
    ingest_status.pack(fill=tk.X)

    sources_label = ttk.Label(
        ingest_tab,
        text="Source Sites (feeds or websites, one URL per line):",
    )
    sources_label.pack(anchor=tk.W)

    sources_box = tk.Text(ingest_tab, height=6, wrap=tk.WORD, font=("Consolas", 10))
    sources_box.pack(fill=tk.X, pady=(4, 8))
    sources_box.insert("1.0", "\n".join(llm.load_source_sites()))

    manual_label = ttk.Label(ingest_tab, text="Manual URL:")
    manual_label.pack(anchor=tk.W)

    manual_row = ttk.Frame(ingest_tab)
    manual_row.pack(fill=tk.X, pady=(4, 8))
    manual_url_var = tk.StringVar()
    manual_url_entry = ttk.Entry(manual_row, textvariable=manual_url_var)
    manual_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    timeout_label = ttk.Label(manual_row, text="Timeout(s):")
    timeout_label.pack(side=tk.LEFT, padx=(8, 4))
    timeout_var = tk.StringVar(value="15")
    timeout_entry = ttk.Entry(manual_row, textvariable=timeout_var, width=6)
    timeout_entry.pack(side=tk.LEFT)

    folder_label = ttk.Label(ingest_tab, text="Local knowledge folder:")
    folder_label.pack(anchor=tk.W)

    folder_row = ttk.Frame(ingest_tab)
    folder_row.pack(fill=tk.X, pady=(4, 8))
    folder_var = tk.StringVar(value=startup_knowledge_folders[0] if startup_knowledge_folders else DEFAULT_KNOWLEDGE_FOLDER)
    folder_entry = ttk.Entry(folder_row, textvariable=folder_var)
    folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    recursive_var = tk.BooleanVar(value=True)
    recursive_check = ttk.Checkbutton(folder_row, text="Recursive", variable=recursive_var)
    recursive_check.pack(side=tk.LEFT, padx=(8, 0))

    def browse_folder() -> None:
        selected = filedialog.askdirectory(title="Select knowledge folder")
        if selected:
            folder_var.set(selected)

    browse_btn = ttk.Button(folder_row, text="Browse", command=browse_folder)
    browse_btn.pack(side=tk.LEFT, padx=(8, 0))

    ingest_log = tk.Text(ingest_tab, wrap=tk.WORD, font=("Consolas", 10), height=14, state=tk.DISABLED)
    ingest_log.pack(fill=tk.BOTH, expand=True)

    ingest_results_queue: queue.Queue[Tuple[str, Dict[str, object]]] = queue.Queue()

    def append_block(speaker: str, text: str) -> None:
        transcript.configure(state=tk.NORMAL)
        transcript.insert(tk.END, f"[{speaker}]\n{text.strip()}\n\n")
        transcript.see(tk.END)
        transcript.configure(state=tk.DISABLED)

    def append_ingest(text: str) -> None:
        ingest_log.configure(state=tk.NORMAL)
        ingest_log.insert(tk.END, f"{text}\n")
        ingest_log.see(tk.END)
        ingest_log.configure(state=tk.DISABLED)

    def set_controls_enabled(enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        input_box.configure(state=state)
        send_btn.configure(state=state)

    def worker_send(user_text: str) -> None:
        answer, metadata = llm.ask_with_metadata(user_text)
        result_queue.put((answer, metadata))

    def _timeout_value() -> int:
        try:
            value = int(timeout_var.get().strip())
            return max(3, min(120, value))
        except Exception:
            return 15

    def worker_ingest_sources(sources: List[str], timeout: int) -> None:
        result = llm.ingest_source_sites(sources=sources, timeout=timeout)
        ingest_results_queue.put(("bulk", result))

    def worker_ingest_manual(url: str, timeout: int) -> None:
        result = llm.ingest_url(url, timeout=timeout)
        ingest_results_queue.put(("manual", result))

    def worker_ingest_folder(folder_path: str, recursive: bool) -> None:
        result = llm.ingest_folder(folder_path=folder_path, recursive=recursive)
        ingest_results_queue.put(("folder", result))

    def poll_results() -> None:
        try:
            answer, metadata = result_queue.get_nowait()
        except queue.Empty:
            root.after(80, poll_results)
            return

        append_block("PERSEUS", answer)
        status_var.set(
            "Ready"
            f" | provider={metadata.get('provider')}"
            f" | quality={metadata.get('quality_score')}"
            f" | refined={metadata.get('refined')}"
            f" | grounded={metadata.get('grounded_with_ingested_context')}"
        )
        set_controls_enabled(True)
        input_box.focus_set()
        root.after(80, poll_results)

    def poll_ingest_results() -> None:
        try:
            kind, payload = ingest_results_queue.get_nowait()
        except queue.Empty:
            root.after(100, poll_ingest_results)
            return

        if kind == "bulk":
            ingest_status_var.set(
                f"Source ingest complete: {payload.get('successes')}/{payload.get('total')} sources succeeded"
            )
            append_ingest(
                (
                    f"Source ingest complete: {payload.get('successes')}/{payload.get('total')} sources succeeded"
                    f" | entries={payload.get('entry_successes', 0)} ok/{payload.get('entry_failures', 0)} fail"
                )
            )
            for item in payload.get("results", []):
                if item.get("ok"):
                    learned = item.get("summary", {}).get("total_items_learned", 0)
                    append_ingest(
                        (
                            f"[OK] {item.get('url')} | title={item.get('title', '')}"
                            f" | type={item.get('source_type', 'source')}"
                            f" | learned={learned}"
                            f" | entries={item.get('entry_successes', 0)}/{item.get('entries_discovered', 0)}"
                        )
                    )
                else:
                    append_ingest(f"[FAIL] {item.get('url')} | error={item.get('error')}")
        elif kind == "manual":
            if payload.get("ok"):
                learned = payload.get("summary", {}).get("total_items_learned", 0)
                ingest_status_var.set("Manual ingest succeeded")
                append_ingest(f"[OK] {payload.get('url')} | title={payload.get('title', '')} | learned={learned}")
            else:
                ingest_status_var.set("Manual ingest failed")
                append_ingest(f"[FAIL] {payload.get('url')} | error={payload.get('error')}")
        else:
            if payload.get("ok"):
                ingest_status_var.set(
                    f"Folder ingest complete: {payload.get('successes')} files learned"
                )
                append_ingest(
                    (
                        f"Folder ingest complete: {payload.get('successes')} ok"
                        f"/{payload.get('failures')} fail"
                        f" | skipped={payload.get('skipped')}"
                        f" | folder={payload.get('folder')}"
                    )
                )
                for item in payload.get("results", [])[:25]:
                    if item.get("ok"):
                        learned = item.get("summary", {}).get("total_items_learned", 0)
                        append_ingest(f"[OK] {item.get('path')} | learned={learned}")
                    else:
                        append_ingest(f"[FAIL] {item.get('path')} | error={item.get('error')}")
            else:
                ingest_status_var.set("Folder ingest failed")
                append_ingest(f"[FAIL] {payload.get('folder')} | error={payload.get('error')}")

        root.after(100, poll_ingest_results)

    def send_message(*_args) -> None:
        user_text = input_var.get().strip()
        if not user_text:
            return

        input_var.set("")
        append_block("YOU", user_text)
        status_var.set("Generating response...")
        set_controls_enabled(False)
        threading.Thread(target=worker_send, args=(user_text,), daemon=True).start()

    def ingest_manual() -> None:
        url = manual_url_var.get().strip()
        if not url:
            return
        timeout = _timeout_value()
        ingest_status_var.set("Manual ingest in progress...")
        append_ingest(f"Manual ingest started: {url} (timeout={timeout}s)")
        threading.Thread(target=worker_ingest_manual, args=(url, timeout), daemon=True).start()

    def ingest_now() -> None:
        raw = sources_box.get("1.0", tk.END)
        sources = [line.strip() for line in raw.splitlines() if line.strip()]
        if not sources:
            ingest_status_var.set("No sources configured")
            return
        saved = llm.save_source_sites(sources)
        if not saved.get("ok"):
            ingest_status_var.set("Source save failed")
            append_ingest(f"[FAIL] source save | error={saved.get('error')}")
            return
        timeout = _timeout_value()
        ingest_status_var.set("Source ingest in progress...")
        append_ingest(f"Source ingest started with {len(sources)} sources (timeout={timeout}s)")
        threading.Thread(target=worker_ingest_sources, args=(sources, timeout), daemon=True).start()

    def save_sources_only() -> None:
        raw = sources_box.get("1.0", tk.END)
        sources = [line.strip() for line in raw.splitlines() if line.strip()]
        saved = llm.save_source_sites(sources)
        if saved.get("ok"):
            ingest_status_var.set(f"Saved {saved.get('count')} source sites")
            append_ingest(f"Saved {saved.get('count')} source sites to {saved.get('path')}")
        else:
            ingest_status_var.set("Source save failed")
            append_ingest(f"[FAIL] source save | error={saved.get('error')}")

    def ingest_folder_now() -> None:
        folder_path = folder_var.get().strip()
        if not folder_path:
            ingest_status_var.set("No folder configured")
            return
        recursive = bool(recursive_var.get())
        ingest_status_var.set("Folder ingest in progress...")
        append_ingest(f"Folder ingest started: {folder_path} (recursive={recursive})")
        threading.Thread(target=worker_ingest_folder, args=(folder_path, recursive), daemon=True).start()

    send_btn = ttk.Button(controls, text="Send", command=send_message)
    send_btn.pack(side=tk.LEFT, padx=(8, 0))

    clear_btn = ttk.Button(controls, text="Clear", command=lambda: transcript.delete("1.0", tk.END))
    clear_btn.pack(side=tk.LEFT, padx=(8, 0))

    ingest_controls = ttk.Frame(ingest_tab)
    ingest_controls.pack(fill=tk.X, pady=(8, 8))

    ingest_now_btn = ttk.Button(ingest_controls, text="Ingest Source Sites Now", command=ingest_now)
    ingest_now_btn.pack(side=tk.LEFT)

    save_sources_btn = ttk.Button(ingest_controls, text="Save Sources", command=save_sources_only)
    save_sources_btn.pack(side=tk.LEFT, padx=(8, 0))

    ingest_manual_btn = ttk.Button(ingest_controls, text="Ingest Manual URL", command=ingest_manual)
    ingest_manual_btn.pack(side=tk.LEFT, padx=(8, 0))

    ingest_folder_btn = ttk.Button(ingest_controls, text="Ingest Folder", command=ingest_folder_now)
    ingest_folder_btn.pack(side=tk.LEFT, padx=(8, 0))

    input_box.bind("<Return>", send_message)
    manual_url_entry.bind("<Return>", lambda _event: ingest_manual())
    folder_entry.bind("<Return>", lambda _event: ingest_folder_now())

    append_block("SYSTEM", "Perseus chat window is ready.")
    append_ingest("Knowledge ingest tab ready.")
    root.after(80, poll_results)
    root.after(100, poll_ingest_results)
    input_box.focus_set()

    if auto_ingest_var.get():
        ingest_now()
    if auto_ingest_folders:
        for folder_path in startup_knowledge_folders:
            append_ingest(f"Auto folder ingest started: {folder_path} (recursive=True)")
            threading.Thread(target=worker_ingest_folder, args=(folder_path, True), daemon=True).start()

    root.after(250, show_donation_popup)

    def on_close() -> None:
        llm.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def launch_portable_llm_terminal_chat(
    db_path: str = "llm_portable_conversations.db",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    strict_local_only: bool = True,
    use_offline_fallback: bool = False,
    knowledge_folders: Optional[List[str]] = None,
    auto_ingest_folders: bool = True,
) -> None:
    """Launch a terminal chat loop with the same learning and knowledge pipeline as the desktop UI."""
    llm = PortableLLM(
        db_path=db_path,
        provider=provider,
        model=model,
        strict_local_only=strict_local_only,
        use_offline_fallback=use_offline_fallback,
    )
    startup_knowledge_folders = knowledge_folders or list(DEFAULT_AUTO_KNOWLEDGE_FOLDERS)

    print(f"Perseus terminal chat ready. Provider={llm.provider} Model={llm.model}")
    print("Commands: /exit, /quit, /stats, /providers, /ingest <folder>")

    try:
        if auto_ingest_folders:
            for folder_path in startup_knowledge_folders:
                print(f"Auto-ingesting local knowledge folder: {folder_path}")
                result = llm.ingest_folder(folder_path=folder_path, recursive=True)
                print(
                    "  "
                    f"{result.get('successes', 0)} learned, "
                    f"{result.get('failures', 0)} failed, "
                    f"{result.get('skipped', 0)} skipped"
                )

        while True:
            try:
                user_text = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_text:
                continue

            command = user_text.lower()
            if command in {"/exit", "/quit", "exit", "quit"}:
                break
            if command == "/stats":
                print(json.dumps(llm.get_stats(), indent=2))
                continue
            if command == "/providers":
                print(", ".join(llm.available_providers()) or "No providers available")
                continue
            if command.startswith("/ingest "):
                folder = user_text.split(" ", 1)[1].strip().strip('"')
                result = llm.ingest_folder(folder_path=folder, recursive=True)
                print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
                continue

            answer, metadata = llm.ask_with_metadata(user_text)
            print("\nPerseus>")
            print(answer)
            print(
                "\n"
                f"[provider={metadata.get('provider')} "
                f"quality={metadata.get('quality_score')} "
                f"grounded={metadata.get('grounded_with_ingested_context')}]"
            )
    finally:
        llm.close()


def _build_folder_index_content(root: Path, files: List[Path], learned_titles: List[str]) -> str:
    """Create a searchable index so folder-level concepts retrieve all learned local files."""
    extension_counts: Dict[str, int] = {}
    for path in files:
        if not path.is_file():
            continue
        extension = path.suffix.lower() or "<no extension>"
        extension_counts[extension] = extension_counts.get(extension, 0) + 1

    learned_list = "\n".join(f"- {title}" for title in learned_titles[:500])
    extension_list = "\n".join(
        f"- {extension}: {count}" for extension, count in sorted(extension_counts.items())
    )
    return (
        f"Folder knowledge index for {root.name}.\n"
        f"This folder is part of the local Perseus learning base and should be considered for relevant answers.\n\n"
        "File type inventory:\n"
        f"{extension_list}\n\n"
        "Learned files:\n"
        f"{learned_list}"
    )


def _extract_user_request(prompt: str) -> str:
    """Recover the original user request from an enriched prompt when present."""
    text = (prompt or "").strip()
    marker = "User request:"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    return text


def _capability_response() -> str:
    """Explain Perseus capabilities when no generative model is available."""
    return (
        "I can act like a local ChatGPT-style assistant: chat normally, explain ideas, help with code, "
        "analyze files, and answer from learned local knowledge. I auto-ingest folders like `knowledge/` "
        "and `Princess protocol/`, including the Princess Protocol docs and SNAP/MACE source, then use that "
        "context when it is relevant. I also store useful chat turns as memory and use self-improvement "
        "signals to make future answers more specific. If Ollama is running, I use the local model for "
        "full generative responses; otherwise this basic fallback handles simple chat and setup guidance."
    )


def _is_capability_prompt(prompt: str) -> bool:
    """Return True for prompts asking what the assistant can do."""
    text = _extract_user_request(prompt)
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .!?\t\r\n")
    capability_markers = [
        "what can you do",
        "what are your capabilities",
        "what are you capable of",
        "tell me what you can do",
        "how can you help",
        "what do you do",
    ]
    return any(marker in lower for marker in capability_markers)


def _is_general_knowledge_prompt(prompt: str) -> bool:
    """Return True for short general-purpose questions that should not require ingested context."""
    text = _extract_user_request(prompt)
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .!?\t\r\n")
    if not lower:
        return False

    if _general_knowledge_fallback(lower):
        return True

    question_starters = (
        "is ",
        "are ",
        "do ",
        "does ",
        "can ",
        "why ",
        "how ",
        "what is ",
        "what are ",
        "explain ",
    )
    domain_markers = [
        "princess protocol",
        "snap",
        "mace",
        "this project",
        "ingested",
        "learned",
        "knowledge base",
        "file",
        "folder",
        "codebase",
    ]
    return lower.startswith(question_starters) and len(lower.split()) <= 12 and not any(
        marker in lower for marker in domain_markers
    )


def _general_knowledge_fallback(prompt: str) -> str:
    """Answer common simple questions directly when no model is available."""
    text = _extract_user_request(prompt)
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .!?\t\r\n")
    lower = re.sub(r"\bhte\b", "the", lower)

    if re.fullmatch(r"is\s+[-+]?\d+(\.\d+)?\s+a\s+number", lower):
        value = re.search(r"[-+]?\d+(\.\d+)?", lower).group(0)
        return (
            f"Yes. {value} is a number because it represents a mathematical quantity. "
            "More specifically, it is a numeral written with digits; in normal arithmetic it can be counted, "
            "compared, added, subtracted, and used in equations."
        )

    if "why is water wet" in lower:
        return (
            "Water is called wet because it sticks to and spreads across many surfaces. At the molecular level, "
            "water molecules are polar, so they attract each other through cohesion and attract other materials "
            "through adhesion. That combination leaves a film of water on things it touches, which we experience "
            "as wetness. Tiny caveat, because science likes being annoying: water itself is often described as the "
            "thing that makes other things wet, while a surface is wet when water adheres to it."
        )

    if "why is the sun hot" in lower or "why sun hot" in lower:
        return (
            "The Sun is hot because its core is powered by nuclear fusion. Enormous gravity compresses hydrogen "
            "so intensely that hydrogen nuclei fuse into helium, releasing energy as radiation. That energy moves "
            "outward through the Sun and eventually escapes as sunlight and heat. The surface is roughly 5,500 C, "
            "while the core is millions of degrees because that is where the fusion is happening."
        )

    if "why do birds fly" in lower or "how do birds fly" in lower:
        return (
            "Birds fly because their bodies are built to generate lift while staying light. Their wings are shaped "
            "so air moves faster over the top than underneath, creating lift; flapping adds thrust; feathers help "
            "control airflow; and hollow bones plus powerful chest muscles reduce weight and provide power. In simple "
            "terms: wings create lift, muscles create motion, and the bird's lightweight body makes flight practical."
        )

    if "why is the sky blue" in lower:
        return (
            "The sky looks blue because sunlight scatters in Earth's atmosphere. Blue light has a shorter wavelength "
            "than red light, so air molecules scatter it more strongly. That scattered blue light reaches your eyes "
            "from all directions, making the sky appear blue during the day."
        )

    if "why is grass green" in lower:
        return (
            "Grass is green because it contains chlorophyll, the pigment plants use to absorb sunlight for photosynthesis. "
            "Chlorophyll absorbs red and blue light strongly but reflects more green light, so our eyes see grass as green."
        )

    if "why is the moon bright" in lower:
        return (
            "The Moon looks bright because it reflects sunlight. It does not make its own light like the Sun; its rocky "
            "surface bounces some sunlight back toward Earth. It can look especially bright because it is close to us, "
            "contrasts strongly with the dark night sky, and sometimes reflects sunlight from a favorable angle."
        )

    if lower in {"is water wet", "is water wet or not"}:
        return (
            "In everyday language, yes, water is wet. More precisely, water makes other materials wet by adhering "
            "to their surfaces and forming a liquid film. So the casual answer is yes; the technical answer is that "
            "wetness describes the interaction between a liquid and a surface."
        )

    return ""


def _heuristic_general_answer(prompt: str) -> str:
    """Give a useful direct answer for simple general questions outside the learned-context path."""
    text = _extract_user_request(prompt)
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .!?\t\r\n")
    lower = re.sub(r"\bhte\b", "the", lower)

    exist_match = re.fullmatch(r"why (?:do|does) (.+?) exist", lower)
    if exist_match:
        subject = exist_match.group(1).strip()
        if subject in {"chairs", "chair"}:
            return (
                "Chairs exist because humans need a stable, comfortable way to rest, work, eat, and gather without "
                "standing or sitting on the ground. They solve a practical body-design problem: our legs and backs get "
                "tired, and raising the body to table/desk height makes tasks easier. Over time, chairs also became "
                "cultural objects - status symbols, design pieces, office tools, and furniture for social spaces."
            )
        return (
            f"{subject.capitalize()} exist because they serve some function, emerge from some cause, or persist because "
            "they are useful enough to be kept around. A good explanation usually asks: what problem do they solve, "
            "what conditions made them possible, and why did they continue instead of disappearing?"
        )

    why_is_match = re.fullmatch(r"why is (.+) ([a-z][a-z-]*)", lower)
    if why_is_match:
        subject = why_is_match.group(1).strip()
        trait = why_is_match.group(2).strip()
        return (
            f"{subject.capitalize()} is {trait} because of its underlying properties and the way it interacts with "
            "its environment. The useful way to break it down is: what it is made of, what forces or processes act on "
            "it, and what effect those processes create."
        )

    why_do_match = re.fullmatch(r"why (?:do|does) (.+?) (.+)", lower)
    if why_do_match:
        subject = why_do_match.group(1).strip()
        action = why_do_match.group(2).strip()
        return (
            f"{subject.capitalize()} {action} because there is usually a mechanism and a payoff behind the behavior: "
            "some physical, biological, social, or practical process makes it happen, and the result solves a problem "
            "or follows naturally from how the thing is built."
        )

    return (
        "Short answer: this is a general question, so the best way to answer is to identify the mechanism, the cause, "
        "and the practical effect. Ask it with a specific subject and I can give a more concrete explanation."
    )


def _is_small_talk_prompt(prompt: str) -> bool:
    """Return True for conversational prompts that should not require learned context."""
    text = _extract_user_request(prompt)
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .!?\t\r\n")
    if not lower:
        return False

    small_talk_markers = [
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how are you today",
        "how's it going",
        "how is it going",
        "what's up",
        "whats up",
        "thank you",
        "thanks",
    ]
    if lower in small_talk_markers:
        return True

    if len(lower.split()) > 8:
        return False

    phrase_markers = [marker for marker in small_talk_markers if " " in marker]
    if any(marker in lower for marker in phrase_markers):
        return True

    single_word_markers = [marker for marker in small_talk_markers if " " not in marker]
    words = set(re.findall(r"[a-z']+", lower))
    return any(marker in words for marker in single_word_markers)


def _read_knowledge_file(path: Path) -> str:
    """Read a supported local knowledge file into plain text."""
    if path.suffix.lower() == ".docx":
        return _extract_docx_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_docx_text(path: Path) -> str:
    """Extract readable text from a Word .docx file without external dependencies."""
    with zipfile.ZipFile(path) as archive:
        try:
            document_xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise RuntimeError("DOCX is missing word/document.xml") from exc

    try:
        root = ET.fromstring(document_xml)
    except ET.ParseError as exc:
        raise RuntimeError(f"Unable to parse DOCX XML: {exc}") from exc

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        parts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def _strip_html_to_text(html: str) -> str:
    """Convert HTML into compact plain text for ingestion."""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_title(html: str) -> str:
    match = re.search(r"(?is)<title>(.*?)</title>", html)
    if not match:
        return ""
    return unescape(match.group(1)).strip()


def fetch_url_text(url: str, timeout: int = 15) -> Dict[str, str]:
    """Fetch URL and return extracted title + plain text."""
    payload = fetch_url_payload(url, timeout=timeout)
    return {"title": payload.get("title", ""), "text": payload.get("text", "")}


def fetch_url_payload(url: str, timeout: int = 15) -> Dict[str, str]:
    """Fetch URL and return raw body plus extracted title/text."""
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = "utf-8"
            content_type = resp.headers.get("Content-Type", "")
            charset_match = re.search(r"charset=([\w-]+)", content_type)
            if charset_match:
                charset = charset_match.group(1)
            html = raw.decode(charset, errors="replace")
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch URL: {exc}") from exc

    title = _extract_title(html)
    text = _strip_html_to_text(html)

    if not text:
        raise RuntimeError("Fetched page but extracted empty text")

    return {"title": title, "text": text, "raw": html, "content_type": content_type}


def _extract_feed_links(raw_text: str, base_url: str) -> List[str]:
    """Extract article links from RSS/Atom feed content."""
    if not raw_text:
        return []

    links: List[str] = []
    seen = set()

    try:
        root = ET.fromstring(raw_text)
    except ET.ParseError:
        return links

    for item in root.findall(".//item"):
        link_node = item.find("link")
        if link_node is not None and link_node.text:
            link = link_node.text.strip()
            if link and link not in seen:
                links.append(link)
                seen.add(link)

    for entry in root.findall(".//{*}entry"):
        for link_node in entry.findall("{*}link"):
            href = (link_node.attrib.get("href") or "").strip()
            if not href:
                continue
            absolute = urljoin(base_url, href)
            if absolute not in seen:
                links.append(absolute)
                seen.add(absolute)

    return links


def _extract_site_links(raw_text: str, base_url: str) -> List[str]:
    """Extract a bounded set of same-site webpage links from normal HTML."""
    if not raw_text:
        return []

    base_host = (urlparse(base_url).netloc or "").lower().removeprefix("www.")
    if not base_host:
        return []

    blocked_extensions = (
        ".7z",
        ".avi",
        ".css",
        ".dmg",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".js",
        ".mov",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".svg",
        ".tar",
        ".webp",
        ".zip",
    )
    links: List[str] = []
    seen = set()

    for match in re.finditer(r"(?is)<a\s+[^>]*href=[\"']([^\"']+)[\"']", raw_text):
        href = unescape(match.group(1)).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        absolute = urljoin(base_url, href)
        absolute = urldefrag(absolute).url
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue

        host = (parsed.netloc or "").lower().removeprefix("www.")
        if host != base_host:
            continue
        if parsed.path.lower().endswith(blocked_extensions):
            continue
        if absolute.rstrip("/") == base_url.rstrip("/"):
            continue
        if absolute in seen:
            continue

        links.append(absolute)
        seen.add(absolute)

    return links
