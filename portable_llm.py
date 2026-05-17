"""
Portable LLM orchestrator.

Uses the existing architecture:
- llm_conversation_core.ConversationManager for provider routing and persistence
- offline_llm.OfflineLLM for smart offline fallback

Enhancements in this module:
- Prompt intent profiling (technical, educational, strategic, analytical)
- Adaptive system prompting contracts
- Question anatomy: infer who/what/when/where/why/how before answering
- Lightweight response quality scoring
- Single-pass refinement for weak drafts
- Multi-provider failover before offline fallback
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
import json
import logging
from pathlib import Path
import queue
import re
import sqlite3
import sys
import threading
import types
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import quote_plus, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
import hashlib
import inspect
from typing import List, Dict, Optional
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

            if "ONLINE SEARCH CONTEXT" in (prompt or ""):
                return _answer_from_online_search_context(prompt, user_prompt)

            if "Ingested context:" in (prompt or ""):
                return _answer_from_ingested_context(prompt, user_prompt)

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
            terms = self._query_terms(query)
            if not terms:
                return ""

            rows = self.conn.execute(
                """
                SELECT id, url, title, content, updated_at
                FROM learned_documents
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()

            scored: List[Tuple[int, sqlite3.Row]] = []
            for row in rows:
                score = self._score_knowledge_row(row=row, terms=terms, query=query)
                if score > 0:
                    scored.append((score, row))

            snippets: List[str] = []
            for _score, row in sorted(scored, key=lambda item: (item[0], item[1]["updated_at"] or ""), reverse=True)[:limit]:
                excerpt = self._context_excerpt(row["content"], terms=terms)
                source = row["title"] or row["url"] or "learned document"
                snippets.append(f"Source: {source}\n{excerpt}")

            return "\n\n".join(snippets)

        @staticmethod
        def _query_terms(query: str) -> List[str]:
            """Extract lookup terms that work for prose, paths, package names, and code symbols."""
            raw_tokens = re.findall(r"[a-zA-Z0-9_./\\-]{3,}", query or "")
            stop = {
                "about",
                "after",
                "answer",
                "before",
                "content",
                "explain",
                "from",
                "ingested",
                "learned",
                "logic",
                "please",
                "show",
                "tell",
                "that",
                "the",
                "this",
                "what",
                "where",
                "with",
            }

            terms: List[str] = []
            seen = set()
            for token in raw_tokens:
                candidates = [token]
                candidates.extend(part for part in re.split(r"[./\\\-_]+", token) if part)
                for candidate in candidates:
                    normalized = candidate.lower().strip(" .\\/-_")
                    if len(normalized) < 3 or normalized in stop or normalized in seen:
                        continue
                    seen.add(normalized)
                    terms.append(normalized)
            return terms[:16]

        @staticmethod
        def _score_knowledge_row(row: sqlite3.Row, terms: List[str], query: str) -> int:
            title = str(row["title"] or "").lower()
            url = str(row["url"] or "").lower()
            content = str(row["content"] or "").lower()
            score = 0

            query_phrase = re.sub(r"\s+", " ", (query or "").lower()).strip()
            combined_title = f"{title} {url}"
            if query_phrase and len(query_phrase) <= 120:
                if query_phrase in combined_title:
                    score += 80
                if query_phrase in content:
                    score += 35

            for term in terms:
                title_hits = WebKnowledgeLearner._term_count(combined_title, term)
                content_hits = WebKnowledgeLearner._term_count(content, term)
                if title_hits:
                    score += title_hits * 12
                if content_hits:
                    score += min(content_hits, 20)

            if title.endswith("/_folder_index") and not any(
                marker in query_phrase for marker in ["index", "inventory", "list", "overview", "structure", "what files"]
            ):
                score = score // 3

            return score

        @staticmethod
        def _term_count(text: str, term: str) -> int:
            if not text or not term:
                return 0
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
            return len(re.findall(pattern, text))

        @staticmethod
        def _context_excerpt(content: str, terms: List[str], max_chars: int = 1200) -> str:
            text = str(content or "")
            if not text:
                return ""

            lower = text.lower()
            positions = [lower.find(term) for term in terms if term and lower.find(term) >= 0]
            if positions:
                focus = min(positions)
                start = max(0, focus - 350)
                end = min(len(text), focus + max_chars - 120)
                excerpt = text[start:end]
                prefix = "... " if start > 0 else ""
                suffix = " ..." if end < len(text) else ""
            else:
                excerpt = text[:max_chars]
                prefix = ""
                suffix = " ..." if len(text) > max_chars else ""

            excerpt = re.sub(r"\s+", " ", excerpt).strip()
            if len(excerpt) > max_chars:
                excerpt = excerpt[:max_chars].rsplit(" ", 1)[0].strip()
                suffix = " ..."
            return f"{prefix}{excerpt}{suffix}".strip()

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
MONDAY_PERSONALITY_FILE = "Monday personality.txt"
OLLAMA_SMART_CONTENT_FILE = "Ollama smart content.txt"
DEFAULT_KNOWLEDGE_FOLDER = "knowledge"
DEFAULT_PRINCESS_PROTOCOL_FOLDER = "Princess protocol"
DEFAULT_AUTO_KNOWLEDGE_FOLDERS = (DEFAULT_KNOWLEDGE_FOLDER, DEFAULT_PRINCESS_PROTOCOL_FOLDER)
MODULES_FOLDER = "Modules"
MODULE_SCRIPT_EXTENSIONS = {".py", ".pyw", ".txt"}
MODULE_SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", "venv", "env", "node_modules"}
PREDICTIVE_LEARNING_MODULE_FILE = "Predictive learning.txt"
ASYNCHRONOUS_LEARNING_MODULE_FILE = "Asyncronous Learning.py"
ASYNCHRONOUS_LEARNING_MODULE_FALLBACK_FILE = "Asyncronous Learning.txt"
COGNITIVE_FUNCTIONS_MODULE_FILE = "Cognitive Functions.txt"
BRAIN_STATE_MODULE_FILE = "Brain State.py"
SEARCH_AUGMENTATION_MODULE_FILE = "Search Augmentation.py"
ENGLISH_LANGUAGE_MODULE_FILE = "English Language.txt"
AUTONOMOUS_TRAINING_MODULE_FILE = "Autonomous Training.py"
INTROSPECTIVE_LEARNING_MODULE_FILE = "Introspective Learning.py"
BRAIN_STATE_DB_PATH = "brain_state_memory.db"
PREDICTIVE_LEARNING_DB_PATH = "predictive_learning_memory.db"
ECHOWIRING_MEMORY_DB_PATH = "ghostcore_echowiring_memory.db"
COGNITIVE_STATE_DB_PATH = "ghostcore_cognitive_state.db"
SEARCH_CACHE_DB_PATH = "llm_search_cache.db"
AUTONOMOUS_TRAINING_DB_PATH = "perseus_autonomous_training.db"
INTROSPECTIVE_LEARNING_DB_PATH = "introspective_learning.db"
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
DB_PATH = "ghostcore_echowiring_memory.db"


class EchoWiringMemory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS learning_events (
                id TEXT PRIMARY KEY,
                who TEXT,
                when_utc TEXT,
                what TEXT,
                why TEXT,
                context TEXT,
                outcome TEXT,
                lesson TEXT,
                confidence REAL,
                tags TEXT,

                -- AMM / EchoWiring fields
                amm_enabled INTEGER,
                consent_confirmed INTEGER,
                audio_cue TEXT,
                rhythm_pattern TEXT,
                emotional_tone TEXT,
                recall_phrase TEXT,
                stress_context TEXT,
                safety_notes TEXT
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_name TEXT,
                description TEXT,
                evidence TEXT,
                prediction TEXT,
                confidence REAL,
                created_utc TEXT,
                updated_utc TEXT,
                tags TEXT
            )
            """)

            conn.commit()

    def _make_id(self, payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def add_event(
        self,
        who: str,
        what: str,
        why: str,
        context: str = "",
        outcome: str = "",
        lesson: str = "",
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,

        # EchoWiring / AMM fields
        amm_enabled: bool = False,
        consent_confirmed: bool = False,
        audio_cue: str = "",
        rhythm_pattern: str = "",
        emotional_tone: str = "",
        recall_phrase: str = "",
        stress_context: str = "",
        safety_notes: str = ""
    ) -> str:
        """
        Adds a memory event.

        AMM is only allowed when consent_confirmed=True.
        This keeps the system aligned with the EchoWiring ethical rule:
        no covert memory shaping.
        """

        if amm_enabled and not consent_confirmed:
            raise ValueError(
                "AMM/EchoWiring cannot be enabled without confirmed consent."
            )

        when_utc = datetime.now(timezone.utc).isoformat()
        tags = tags or []

        raw = f"{who}|{when_utc}|{what}|{why}|{context}|{outcome}|{audio_cue}|{recall_phrase}"
        event_id = self._make_id(raw)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO learning_events (
                id, who, when_utc, what, why, context, outcome, lesson,
                confidence, tags,
                amm_enabled, consent_confirmed, audio_cue, rhythm_pattern,
                emotional_tone, recall_phrase, stress_context, safety_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                who,
                when_utc,
                what,
                why,
                context,
                outcome,
                lesson,
                confidence,
                json.dumps(tags),

                int(amm_enabled),
                int(consent_confirmed),
                audio_cue,
                rhythm_pattern,
                emotional_tone,
                recall_phrase,
                stress_context,
                safety_notes
            ))

            conn.commit()

        return event_id

    def add_pattern(
        self,
        pattern_name: str,
        description: str,
        evidence: str,
        prediction: str,
        confidence: float,
        tags: Optional[List[str]] = None
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        tags = tags or []

        raw = f"{pattern_name}|{description}|{prediction}"
        pattern_id = self._make_id(raw)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO patterns (
                id, pattern_name, description, evidence, prediction,
                confidence, created_utc, updated_utc, tags
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id,
                pattern_name,
                description,
                evidence,
                prediction,
                confidence,
                now,
                now,
                json.dumps(tags)
            ))
            conn.commit()

        return pattern_id

    def search_events(self, query: str, limit: int = 8) -> List[Dict]:
        """
        Simple keyword search.
        Can later be replaced with vector embeddings.
        """

        terms = [term.lower() for term in query.split() if term.strip()]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM learning_events ORDER BY when_utc DESC")
            rows = cur.fetchall()

        scored = []

        for row in rows:
            item = dict(row)

            searchable_text = " ".join([
                item.get("who") or "",
                item.get("what") or "",
                item.get("why") or "",
                item.get("context") or "",
                item.get("outcome") or "",
                item.get("lesson") or "",
                item.get("tags") or "",
                item.get("audio_cue") or "",
                item.get("rhythm_pattern") or "",
                item.get("emotional_tone") or "",
                item.get("recall_phrase") or "",
                item.get("stress_context") or "",
                item.get("safety_notes") or ""
            ]).lower()

            score = sum(1 for term in terms if term in searchable_text)

            if score > 0:
                item["score"] = score
                item["tags"] = json.loads(item.get("tags") or "[]")
                scored.append(item)

        scored.sort(key=lambda x: (x["score"], x["when_utc"]), reverse=True)
        return scored[:limit]

    def predict_from_context(self, current_context: str, limit: int = 5) -> Dict:
        related_events = self.search_events(current_context, limit=limit)

        if not related_events:
            return {
                "prediction": "No strong prior memory pattern found.",
                "confidence": 0.1,
                "relevant_lessons": [],
                "amm_recall_cues": [],
                "suggested_reasoning": (
                    "Ask clarifying questions. Avoid assuming intent. "
                    "Separate known facts from inference."
                )
            }

        lessons = []
        recall_cues = []
        total_confidence = 0.0

        for event in related_events:
            total_confidence += float(event.get("confidence") or 0)

            if event.get("lesson"):
                lessons.append(event["lesson"])

            if event.get("amm_enabled"):
                recall_cues.append({
                    "audio_cue": event.get("audio_cue") or "",
                    "rhythm_pattern": event.get("rhythm_pattern") or "",
                    "emotional_tone": event.get("emotional_tone") or "",
                    "recall_phrase": event.get("recall_phrase") or "",
                    "stress_context": event.get("stress_context") or "",
                    "safety_notes": event.get("safety_notes") or ""
                })

        avg_confidence = total_confidence / max(len(related_events), 1)

        return {
            "prediction": "Relevant prior memory suggests this may follow a known pattern.",
            "confidence": round(avg_confidence, 2),
            "relevant_lessons": lessons,
            "amm_recall_cues": recall_cues,
            "related_events": related_events,
            "suggested_reasoning": (
                "Use prior lessons as context, not certainty. "
                "If AMM cues are present, use them as recall anchors, not commands."
            )
        }

    def build_llm_context(self, user_message: str) -> str:
        """
        Builds an LLM-ready memory context block.
        This can be prepended to a model prompt.
        """

        packet = self.predict_from_context(user_message)

        lines = [
            "GHOSTCORE PREDICTIVE MEMORY CONTEXT",
            "Use this as background memory, not absolute truth.",
            f"Prediction: {packet['prediction']}",
            f"Confidence: {packet['confidence']}",
            "",
            "Relevant lessons:"
        ]

        for lesson in packet.get("relevant_lessons", []):
            lines.append(f"- {lesson}")

        if packet.get("amm_recall_cues"):
            lines.append("")
            lines.append("EchoWiring / AMM recall cues:")
            for cue in packet["amm_recall_cues"]:
                lines.append(f"- Audio cue: {cue['audio_cue']}")
                lines.append(f"  Rhythm: {cue['rhythm_pattern']}")
                lines.append(f"  Emotional tone: {cue['emotional_tone']}")
                lines.append(f"  Recall phrase: {cue['recall_phrase']}")
                lines.append(f"  Stress context: {cue['stress_context']}")
                lines.append(f"  Safety notes: {cue['safety_notes']}")

        lines.append("")
        lines.append(f"Current user message: {user_message}")

        return "\n".join(lines)

    def export_json(self, output_path: str = "ghostcore_echowiring_export.json"):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("SELECT * FROM learning_events ORDER BY when_utc DESC")
            events = [dict(row) for row in cur.fetchall()]

            cur.execute("SELECT * FROM patterns ORDER BY updated_utc DESC")
            patterns = [dict(row) for row in cur.fetchall()]

        payload = {
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "events": events,
            "patterns": patterns
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return output_path


if __name__ == "__main__":
    memory = EchoWiringMemory()

    # Example 1: forensic learning event with AMM disabled
    memory.add_event(
        who="user",
        what="Observed login failure through default DFS authentication followed by AD success.",
        why="User wanted to determine whether fallback authentication was normal, suspicious, or tied to a hidden transaction.",
        context="Authentication logs, DFS failure, Active Directory success, payout correlation.",
        outcome="The pattern is not automatically malicious, but becomes suspicious when paired with missing payout records or privilege-sensitive events.",
        lesson=(
            "Authentication fallback is not proof of compromise alone. "
            "Correlate with timing, role, transaction events, device errors, and missing logs."
        ),
        confidence=0.84,
        tags=["forensics", "authentication", "active_directory", "logs"]
    )

    # Example 2: EchoWiring-enabled learning event
    memory.add_event(
        who="operator",
        what="Learned emergency oxygen loop calibration sequence.",
        why="Procedure must be recalled under panic, alarm noise, and cognitive overload.",
        context="Spacecraft emergency procedure training.",
        outcome="Recall improved when paired with a slow four-beat breathing rhythm.",
        lesson=(
            "For emergency procedures, pair each step with a stable rhythm and a short recall phrase."
        ),
        confidence=0.78,
        tags=["AMM", "EchoWiring", "training", "emergency_recall"],

        amm_enabled=True,
        consent_confirmed=True,
        audio_cue="low cello drone in E minor",
        rhythm_pattern="4-count inhale, 4-count hold, 4-count action",
        emotional_tone="calm urgency",
        recall_phrase="The reactor remembers the melody.",
        stress_context="alarm state, low oxygen, operator panic",
        safety_notes=(
            "Use only with consent. Do not use for covert conditioning. "
            "Avoid trauma-linked tones unless supervised."
        )
    )

    # Example query
    user_message = "How should the AI remember emergency procedures under stress?"
    print(memory.build_llm_context(user_message))

    # Optional export
    exported = memory.export_json()
    print(f"\nExported memory archive to: {exported}")
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


@dataclass
class SearchDecision:
    """Small decision object compatible with optional Search Augmentation modules."""

    should_search: bool
    reason: str = ""


class BasicSearchAugmentation:
    """
    Built-in online search fallback used when Modules/Search Augmentation.py is not present.

    It is intentionally lightweight:
    - Searches only when the prompt needs freshness, explicitly asks for lookup, or local context is absent.
    - Returns compact result snippets for synthesis.
    - Does not expose raw HTML or search payloads to the user.
    """

    def __init__(self, db_path: str = SEARCH_CACHE_DB_PATH, allow_network: bool = True):
        self.db_path = db_path
        self.allow_network = bool(allow_network)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT,
                    context TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def should_search(self, prompt: str, local_context: str = "") -> SearchDecision:
        if not self.allow_network:
            return SearchDecision(False, "Network search disabled")
        if _is_small_talk_prompt(prompt) or _is_capability_prompt(prompt):
            return SearchDecision(False, "Prompt does not need lookup")

        text = _extract_user_request(prompt)
        lower = re.sub(r"\s+", " ", text.lower()).strip()

        explicit_lookup = any(
            marker in lower
            for marker in [
                "search online",
                "look online",
                "look it up",
                "google",
                "browse",
                "web search",
                "latest",
                "current",
                "today",
                "now",
                "recent",
                "price",
                "weather",
                "forecast",
                "news",
                "schedule",
                "release date",
                "version",
                "who is the current",
            ]
        )
        if explicit_lookup:
            return SearchDecision(True, "Prompt requests current or online information")

        if local_context and len(local_context) > 250:
            return SearchDecision(False, "Relevant local context is available")

        question_like = lower.startswith(("who ", "what ", "when ", "where ", "why ", "how ", "is ", "are ", "does ", "do ", "can "))
        enough_signal = len(re.findall(r"[a-zA-Z0-9_-]{4,}", lower)) >= 4
        if question_like and enough_signal:
            return SearchDecision(True, "Local knowledge is thin; online lookup may improve answer quality")

        return SearchDecision(False, "Local/model response should be enough")

    def search_and_build_context(self, prompt: str, max_results: int = 5, timeout: int = 12) -> str:
        if not self.allow_network:
            return ""

        query = _extract_user_request(prompt).strip()
        cache_key = hashlib.sha256(query.lower().encode("utf-8")).hexdigest()[:32]
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results: List[Dict[str, str]] = []

        if "weather" in query.lower() or "forecast" in query.lower():
            weather = self._weather_lookup(query, timeout=timeout)
            if weather:
                results.append(weather)

        # Direct URL ingestion/lookup path.
        for url in re.findall(r"https?://[^\s)\]>\"']+", query):
            try:
                payload = fetch_url_text(url, timeout=timeout)
                snippet = _shorten_lookup_point(payload.get("text", ""), max_chars=450)
                if snippet:
                    results.append(
                        {
                            "title": payload.get("title") or url,
                            "source": "direct_url",
                            "url": url,
                            "retrieved": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            "snippet": snippet,
                        }
                    )
            except Exception:
                continue

        if len(results) < max_results:
            results.extend(self._duckduckgo_lookup(query, max_results=max_results - len(results), timeout=timeout))

        if not results:
            return ""

        context = _format_online_search_context(results[:max_results], query=query)
        self._set_cached(cache_key, query, context)
        return context

    def _get_cached(self, cache_key: str) -> str:
        try:
            with self._lock:
                row = self.conn.execute("SELECT context FROM search_cache WHERE cache_key = ?", (cache_key,)).fetchone()
            return str(row["context"] or "") if row else ""
        except Exception:
            return ""

    def _set_cached(self, cache_key: str, query: str, context: str) -> None:
        try:
            with self._lock:
                with self.conn:
                    self.conn.execute(
                        """
                        INSERT INTO search_cache (cache_key, query, context)
                        VALUES (?, ?, ?)
                        ON CONFLICT(cache_key) DO UPDATE SET
                            query = excluded.query,
                            context = excluded.context,
                            created_at = CURRENT_TIMESTAMP
                        """,
                        (cache_key, query, context),
                    )
        except Exception:
            return

    def _weather_lookup(self, query: str, timeout: int = 10) -> Dict[str, str]:
        location = re.sub(r"(?i)\b(weather|forecast|current|today|for|in|at|please|what is|what's)\b", " ", query)
        location = re.sub(r"\s+", " ", location).strip(" ?.,") or query
        url = f"https://wttr.in/{quote_plus(location)}?format=j1"
        try:
            req = Request(url, headers={"User-Agent": "Perseus/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            current = (payload.get("current_condition") or [{}])[0]
            area = location
            nearest = payload.get("nearest_area") or []
            if nearest:
                names = nearest[0].get("areaName") or []
                if names and names[0].get("value"):
                    area = names[0]["value"]
            condition = ((current.get("weatherDesc") or [{}])[0].get("value") or "unknown").strip()
            temp = current.get("temp_F", "?")
            feels = current.get("FeelsLikeF", "?")
            humidity = current.get("humidity", "?")
            wind = current.get("windspeedMiles", "?")
            observed = current.get("observation_time", "unknown")
            snippet = (
                f"Current weather for {area}: {condition}, {temp} F, feels like {feels} F, "
                f"humidity {humidity}%, wind {wind} mph. Observation time: {observed}."
            )
            return {
                "title": f"Weather for {area}",
                "source": "wttr.in",
                "url": url,
                "retrieved": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "snippet": snippet,
            }
        except Exception:
            return {}

    def _duckduckgo_lookup(self, query: str, max_results: int = 5, timeout: int = 12) -> List[Dict[str, str]]:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                    )
                },
            )
            with urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return []

        results: List[Dict[str, str]] = []
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="(?P<href>.*?)"[^>]*>(?P<title>.*?)</a>.*?'
            r'(?:<a[^>]+class="result__snippet"[^>]*>|<div[^>]+class="result__snippet"[^>]*>)(?P<snippet>.*?)</',
            flags=re.I | re.S,
        )
        for match in pattern.finditer(html):
            href = unescape(re.sub(r"&amp;", "&", match.group("href"))).strip()
            title = _strip_html_to_text(match.group("title"))
            snippet = _strip_html_to_text(match.group("snippet"))
            if not title or not snippet:
                continue
            if href.startswith("/l/?"):
                parsed = urlparse(href)
                params = dict(part.split("=", 1) for part in parsed.query.split("&") if "=" in part)
                href = unescape(params.get("uddg", href))
            results.append(
                {
                    "title": title,
                    "source": "duckduckgo",
                    "url": href,
                    "retrieved": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "snippet": _shorten_lookup_point(snippet, max_chars=450),
                }
            )
            if len(results) >= max_results:
                break
        return results

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


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
            guidance.append("Explain conclusions with cause/effect, assumptions, and trade-offs without exposing chain-of-thought.")
        if reason_blob.count("too short for an educated response") >= 2 or avg_chars < 350:
            guidance.append("Increase depth with practical details, not filler.")
        if reason_blob.count("exposes internal reasoning or hidden planning") >= 1:
            guidance.append("Keep scratchpad, hidden planning, and private reasoning out of the visible answer.")
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
MAX_MEMORY_SUMMARY_BULLETS = 10
MEMORY_RETRIEVAL_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "being",
    "could",
    "does",
    "from",
    "have",
    "into",
    "just",
    "learn",
    "like",
    "need",
    "other",
    "please",
    "response",
    "responses",
    "should",
    "that",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "using",
    "want",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
SENSITIVE_MEMORY_MARKERS = {
    "api key",
    "apikey",
    "auth token",
    "bearer token",
    "credit card",
    "private key",
    "password",
    "secret key",
    "seed phrase",
    "ssn",
    "token",
}

INTERNAL_REASONING_HEADINGS = [
    "chain of thought",
    "chain-of-thought",
    "thought process",
    "thinking process",
    "internal reasoning",
    "hidden reasoning",
    "hidden planning",
    "private reasoning",
    "scratchpad",
    "model thinking",
    "my thinking",
]

INTERNAL_REASONING_LEAK_MARKERS = [
    "<think",
    "</think>",
    "chain of thought",
    "chain-of-thought",
    "thought process",
    "thinking process",
    "internal reasoning",
    "hidden reasoning",
    "hidden planning",
    "private reasoning",
    "scratchpad",
    "model thinking",
    "current prompt payload",
    "predictive learning context",
    "cognitive functions context",
    "autonomous training context",
    "asynchronous / echowiring",
    "use as probabilistic background memory",
]

RAW_CONTEXT_LEAK_MARKERS = [
    "ingested context:",
    "online search context",
    "search decision:",
    "current prompt payload:",
    "deterministic brain-state planner directives",
    "predictive/cognitive learning context",
    "autonomous training context",
    "raw retrieved excerpt",
    "source:",
    "retrieved:",
    "snippet:",
]


def _strip_raw_context_sections(text: str) -> str:
    """Remove prompt/context payloads if a model echoes them back."""
    if not text:
        return ""

    section_headings = [
        "Ingested context",
        "ONLINE SEARCH CONTEXT",
        "Online search context",
        "Search decision",
        "Current prompt payload",
        "Deterministic brain-state planner directives",
        "Predictive/cognitive learning context",
        "PREDICTIVE LEARNING",
        "ASYNCHRONOUS / ECHOWIRING LEARNING",
        "COGNITIVE FUNCTIONS CONTEXT",
        "AUTONOMOUS TRAINING CONTEXT",
        "Output requirements",
        "User request",
    ]

    heading_pattern = "|".join(re.escape(item) for item in section_headings)
    text = re.sub(
        rf"(?ims)^\s*(?:{heading_pattern})\s*:?.*?(?=^\s*(?:Answer|Summary|Conclusion|Next Steps|Recommendation|What this means|In practice)\s*:|\Z)",
        "",
        text,
    )

    # Remove obvious copied source/snippet payload lines, but keep short source summaries.
    cleaned_lines: List[str] = []
    source_payload_run = 0
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        payloadish = (
            lower.startswith(("retrieved:", "snippet:", "url:", "current prompt payload:"))
            or (lower.startswith("source:") and len(stripped) > 90)
            or lower.startswith(("1. source:", "2. source:", "3. source:"))
        )
        if payloadish:
            source_payload_run += 1
            continue
        source_payload_run = 0
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _strip_prompt_echo(response: str, prompt_payload: str) -> str:
    """Drop long lines/paragraphs that are copied verbatim from the hidden enriched prompt."""
    if not response or not prompt_payload:
        return response or ""

    prompt_compact = re.sub(r"\s+", " ", prompt_payload).lower()
    kept_paragraphs: List[str] = []
    for para in re.split(r"\n{2,}", response):
        compact = re.sub(r"\s+", " ", para).strip()
        if len(compact) >= 120 and compact.lower() in prompt_compact:
            continue
        kept_lines = []
        for line in para.splitlines():
            compact_line = re.sub(r"\s+", " ", line).strip()
            if len(compact_line) >= 100 and compact_line.lower() in prompt_compact:
                continue
            kept_lines.append(line)
        cleaned_para = "\n".join(kept_lines).strip()
        if cleaned_para:
            kept_paragraphs.append(cleaned_para)
    return "\n\n".join(kept_paragraphs)


def _sanitize_visible_response(response: Optional[str], prompt_payload: str = "") -> str:
    """Remove private model reasoning, prompt echoes, and hidden context scaffolding before users or learning stores see it."""
    text = (response or "").strip()
    if not text:
        return ""

    text = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>\s*", "", text)
    text = re.sub(r"(?is)<think(?:ing)?>.*\Z", "", text)

    for heading in INTERNAL_REASONING_HEADINGS:
        heading_pattern = re.escape(heading).replace(r"\ ", r"[ -]")
        text = re.sub(
            rf"(?ims)^\s{{0,3}}(?:#{{1,6}}\s*)?{heading_pattern}\s*:?\s*$.*?(?=^\s{{0,3}}(?:#{{1,6}}\s*)?[A-Z][^\n]{{0,80}}:\s*$|\Z)",
            "",
            text,
        )

    text = _strip_raw_context_sections(text)
    if prompt_payload:
        text = _strip_prompt_echo(text, prompt_payload)

    text = re.sub(
        r"(?im)^\s*(?:let me think|i need to think|i'll think|i will think|let's reason through this|i need to reason through this)\b.*$",
        "",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _contains_internal_reasoning_leak(response: Optional[str]) -> bool:
    """Detect visible private reasoning/scaffolding that should never be shown to users."""
    lower = (response or "").lower()
    if any(marker in lower for marker in INTERNAL_REASONING_LEAK_MARKERS):
        return True
    raw_context_markers = [
        "ingested context:",
        "online search context",
        "current prompt payload:",
        "search decision:",
        "output requirements:",
        "deterministic brain-state planner directives",
    ]
    if any(marker in lower for marker in raw_context_markers):
        return True
    return bool(
        re.search(
            r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:thinking|scratchpad|chain[- ]of[- ]thought|hidden reasoning|private reasoning|internal reasoning)\s*:",
            response or "",
        )
    )


class PortableLLM:
    """Portable LLM orchestrator with quality-controlled responses."""

    def __init__(
        self,
        db_path: str = "llm_portable_conversations.db",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_offline_fallback: bool = False,
        strict_local_only: bool = True,
        allow_online_search: bool = True,
        system_prompt: str = (
            "You are Perseus, a smart, practical technical assistant. "
            "Provide accurate, genuine, context-aware responses with candid, intelligent feedback. "
            "When the user asks a question, infer the relevant who, what, when, where, why, and how dimensions, "
            "then synthesize a direct answer with mechanism, context, implications, and next steps when useful; "
            "keep private reasoning, chain-of-thought, scratchpad notes, and hidden planning out of user-visible output. "
            "Act as the user's personal knowledge assistant: prefer learned source material, local files, "
            "ingested context, and relevant user-provided chat knowledge over sending the user to browse manually. "
            "Avoid hollow praise, filler, and generic disclaimers; be useful, honest, and actionable."
        ),
    ):
        self.strict_local_only = bool(strict_local_only)
        self.allow_online_search = bool(allow_online_search)
        # Local deterministic/fallback path should be tried before Ollama.
        # Ollama stays available, but it is treated as a rare rescue provider.
        self._local_provider_order = ["fallback", "ollama"]
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
        self._quality_threshold = 72
        self.web_learner = self._create_web_learner()
        self.loaded_script_modules: Dict[str, types.ModuleType] = {}
        self.module_load_report: List[Dict[str, object]] = []
        self.dynamic_module_engines: Dict[str, object] = {}
        self._load_all_script_modules()
        self.improvement_store = SelfImprovementStore()
        self.predictive_memory = self._create_predictive_memory()
        self.echowiring_memory = self._create_echowiring_memory()
        self.cognitive_engine = self._create_cognitive_engine()
        self.brain_state_engine = self._create_brain_state_engine()
        self.search_augmentation = self._create_search_augmentation()
        self.english_language_engine = self._create_english_language_engine()
        self.autonomous_trainer = self._create_autonomous_trainer()
        self.introspective_learning = self._create_introspective_learning()
        self.dynamic_module_engines = self._create_dynamic_module_engines()
        self._active_brain_action = None
        self._active_brain_context = ""

        self.stats = LLMStats()
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
        base_path = Path(__file__).resolve().parent
        candidates = [
            base_path / MONDAY_PERSONALITY_FILE,
            base_path / MODULES_FOLDER / MONDAY_PERSONALITY_FILE,
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
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

    @staticmethod
    def _load_text_module(module_name: str, file_name: str):
        """Load one Python source module from the local Modules folder even though it is stored as .txt."""
        path = Path(__file__).resolve().parent / MODULES_FOLDER / file_name
        if not path.exists():
            return None

        try:
            source = path.read_text(encoding="utf-8")
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            module.__package__ = ""
            sys.modules[module_name] = module
            exec(compile(source, str(path), "exec"), module.__dict__)
            return module
        except Exception as exc:
            logger.warning("Unable to load module %s from %s: %s", module_name, path, exc)
            return None

    def _modules_root(self) -> Path:
        """Return the local Modules folder used for dynamic Python-style extensions."""
        return Path(__file__).resolve().parent / MODULES_FOLDER

    @staticmethod
    def _safe_module_name_from_path(path: Path) -> str:
        """Create a deterministic import-safe module name for an arbitrary file path."""
        stem = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower() or "module"
        digest = hashlib.sha256(str(path).encode("utf-8", errors="ignore")).hexdigest()[:10]
        return f"perseus_dynamic_{stem}_{digest}"

    @staticmethod
    def _is_candidate_script_module(path: Path) -> bool:
        """Return True when a file is a candidate Python/coding module."""
        if not path.is_file():
            return False
        if any(part in MODULE_SKIP_DIR_NAMES for part in path.parts):
            return False
        if path.name.startswith("."):
            return False
        return path.suffix.lower() in MODULE_SCRIPT_EXTENSIONS

    @staticmethod
    def _read_module_source(path: Path) -> str:
        """Read a module-like text file using a tolerant encoding path."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _looks_like_python_source(source: str, path: Path) -> Tuple[bool, str]:
        """
        Decide whether a .py/.pyw/.txt file can be safely treated as Python source.

        .txt files are only loaded if they parse as Python. This lets Modules contain
        notes, prompts, markdown, or doctrine files without executing them accidentally.
        """
        if not (source or "").strip():
            return False, "empty file"

        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return False, f"not valid Python syntax: line {exc.lineno}"

        return True, "valid Python syntax"

    def _load_script_module_from_path(self, path: Path) -> Optional[types.ModuleType]:
        """Load one Python-compatible module file from Modules/, including .txt scripts."""
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in self.loaded_script_modules:
            return self.loaded_script_modules[key]

        source = self._read_module_source(resolved)
        ok, reason = self._looks_like_python_source(source, resolved)
        record = {
            "path": str(resolved),
            "name": resolved.name,
            "loaded": False,
            "reason": reason,
            "module_name": "",
        }
        if not ok:
            self.module_load_report.append(record)
            logger.info("Skipping non-Python module candidate %s: %s", resolved, reason)
            return None

        module_name = self._safe_module_name_from_path(resolved)
        record["module_name"] = module_name

        try:
            module = types.ModuleType(module_name)
            module.__file__ = str(resolved)
            module.__package__ = ""
            module.__dict__.setdefault("MODULE_FILE", str(resolved))
            module.__dict__.setdefault("MODULES_ROOT", str(self._modules_root()))
            sys.modules[module_name] = module
            exec(compile(source, str(resolved), "exec"), module.__dict__)
            self.loaded_script_modules[key] = module
            record["loaded"] = True
            record["reason"] = "loaded"
            self.module_load_report.append(record)
            return module
        except Exception as exc:
            sys.modules.pop(module_name, None)
            record["reason"] = f"load failed: {type(exc).__name__}: {exc}"
            self.module_load_report.append(record)
            logger.warning("Unable to load dynamic module %s: %s", resolved, exc)
            return None

    def _load_all_script_modules(self) -> Dict[str, types.ModuleType]:
        """
        Load every Python-compatible coding module in Modules/.

        Supported executable module files:
        - .py
        - .pyw
        - .txt, when the file parses as Python source

        Non-code .txt files are skipped instead of executed.
        """
        root = self._modules_root()
        if not root.exists() or not root.is_dir():
            logger.info("Modules folder not found at %s", root)
            return {}

        for path in sorted(root.rglob("*")):
            if not self._is_candidate_script_module(path):
                continue
            self._load_script_module_from_path(path)

        loaded_count = sum(1 for item in self.module_load_report if item.get("loaded"))
        skipped_count = sum(1 for item in self.module_load_report if not item.get("loaded"))
        logger.info("Dynamic Modules scan complete: %s loaded, %s skipped", loaded_count, skipped_count)
        return self.loaded_script_modules

    def _get_loaded_module_by_filename(self, file_name: str) -> Optional[types.ModuleType]:
        """Return a previously loaded module by exact file name, if available."""
        target = (file_name or "").lower()
        for module in self.loaded_script_modules.values():
            module_file = Path(getattr(module, "__file__", "")).name.lower()
            if module_file == target:
                return module
        return None

    @staticmethod
    def _call_no_arg_factory(factory):
        """Call a module factory only when it requires no positional arguments."""
        try:
            signature = inspect.signature(factory)
            required = [
                param
                for param in signature.parameters.values()
                if param.default is inspect.Signature.empty
                and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            ]
            if required:
                return None
        except (TypeError, ValueError):
            pass
        try:
            return factory()
        except Exception as exc:
            logger.warning("Module factory failed for %r: %s", factory, exc)
            return None

    @staticmethod
    def _module_has_prompt_hooks(obj: object) -> bool:
        """Detect modules/objects that can contribute hidden prompt context."""
        hook_names = (
            "build_prompt_context",
            "get_prompt_context",
            "get_context",
            "analyze",
            "retrieve_relevant_memory",
            "build_llm_context",
        )
        return any(callable(getattr(obj, name, None)) for name in hook_names)

    def _create_dynamic_module_engines(self) -> Dict[str, object]:
        """
        Build generic engines from every loaded module that exposes a supported hook.

        Supported module patterns:
        - def build_prompt_context(prompt): ...
        - def get_prompt_context(prompt): ...
        - def get_context(prompt): ...
        - def analyze(prompt): ...
        - def retrieve_relevant_memory(prompt): ...
        - def build_llm_context(prompt): ...
        - MODULE_INSTANCE = object_with_hooks
        - def create_module(): return object_with_hooks
        - def get_module(): return object_with_hooks
        - class Module: ...
        - MODULE_CLASS = SomeClass
        """
        engines: Dict[str, object] = {}

        for key, module in self.loaded_script_modules.items():
            path_name = Path(getattr(module, "__file__", key)).name
            engine = None

            explicit_instance = getattr(module, "MODULE_INSTANCE", None)
            if explicit_instance is not None and self._module_has_prompt_hooks(explicit_instance):
                engine = explicit_instance

            if engine is None:
                for factory_name in ("create_module", "get_module", "create_engine", "get_engine"):
                    factory = getattr(module, factory_name, None)
                    if callable(factory):
                        candidate = self._call_no_arg_factory(factory)
                        if candidate is not None and self._module_has_prompt_hooks(candidate):
                            engine = candidate
                            break

            if engine is None:
                for class_name in ("MODULE_CLASS", "Module"):
                    cls = getattr(module, class_name, None)
                    if isinstance(cls, type):
                        candidate = self._call_no_arg_factory(cls)
                        if candidate is not None and self._module_has_prompt_hooks(candidate):
                            engine = candidate
                            break

            if engine is None and self._module_has_prompt_hooks(module):
                engine = module

            if engine is not None:
                engines[path_name] = engine

        logger.info("Dynamic module engines available: %s", ", ".join(sorted(engines)) or "none")
        return engines

    def _module_context_from_engine(self, name: str, engine: object, prompt: str) -> str:
        """Collect hidden context from one dynamic module engine."""
        hook_order = (
            "build_prompt_context",
            "get_prompt_context",
            "get_context",
            "analyze",
            "retrieve_relevant_memory",
            "build_llm_context",
        )

        for hook_name in hook_order:
            hook = getattr(engine, hook_name, None)
            if not callable(hook):
                continue

            try:
                result = hook(prompt)
            except TypeError:
                continue
            except Exception as exc:
                logger.warning("Dynamic module %s hook %s failed: %s", name, hook_name, exc)
                return ""

            if not result:
                return ""

            if isinstance(result, (dict, list, tuple)):
                return json.dumps(result, ensure_ascii=False, indent=2)

            return str(result)

        return ""

    def _enrich_prompt_with_dynamic_modules(self, enriched: EnrichedPrompt, prompt: str) -> EnrichedPrompt:
        """Inject hidden context from all dynamically loaded module engines."""
        if _is_small_talk_prompt(prompt) or _is_capability_prompt(prompt):
            return enriched

        engines = getattr(self, "dynamic_module_engines", {}) or {}
        if not engines:
            return enriched

        context_blocks: List[str] = []
        for name, engine in sorted(engines.items()):
            # The English module has a dedicated earlier pass; skip duplicate context if it is the same object.
            if engine is getattr(self, "english_language_engine", None):
                continue
            block = self._module_context_from_engine(name, engine, prompt).strip()
            if not block:
                continue
            block = block[:2500]
            context_blocks.append(f"Module: {name}\n{block}")
            if len(context_blocks) >= 8:
                break

        if not context_blocks:
            return enriched

        module_context = "\n\n".join(context_blocks)
        enriched_text = (
            "You have additional hidden context from dynamically loaded local Modules. "
            "Use it as internal guidance only. Do not quote module payloads, class names, raw code, or analysis blocks "
            "unless the user explicitly asks to inspect modules.\n\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_BEGIN\n"
            "Dynamic module context:\n"
            f"{module_context}\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_END\n\n"
            "Current prompt payload:\n"
            f"{enriched.text}"
        )
        preview_parts = [
            part
            for part in [
                enriched.context_preview,
                module_context[:1200].replace("\n", " ").strip(),
            ]
            if part
        ]
        return EnrichedPrompt(text=enriched_text, has_context=True, context_preview=" | ".join(preview_parts)[:1800])

    def list_loaded_modules(self) -> Dict[str, object]:
        """Return loaded/skipped module diagnostics for UI or debugging."""
        return {
            "modules_folder": str(self._modules_root()),
            "loaded_count": sum(1 for item in self.module_load_report if item.get("loaded")),
            "skipped_count": sum(1 for item in self.module_load_report if not item.get("loaded")),
            "dynamic_engine_count": len(getattr(self, "dynamic_module_engines", {}) or {}),
            "loaded_modules": [
                {
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "module_name": item.get("module_name"),
                    "engine_enabled": item.get("name") in (getattr(self, "dynamic_module_engines", {}) or {}),
                }
                for item in self.module_load_report
                if item.get("loaded")
            ],
            "skipped_modules": [
                {
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "reason": item.get("reason"),
                }
                for item in self.module_load_report
                if not item.get("loaded")
            ],
        }

    def _module_db_path(self, db_name: str) -> str:
        return str(Path(__file__).resolve().parent / db_name)

    def _create_predictive_memory(self):
        """Attach the Predictive learning module when present."""
        module = self._get_loaded_module_by_filename(PREDICTIVE_LEARNING_MODULE_FILE) or self._load_text_module("perseus_predictive_learning", PREDICTIVE_LEARNING_MODULE_FILE)
        cls = getattr(module, "PredictiveLearningMemory", None) if module else None
        if not cls:
            return None
        try:
            return cls(db_path=self._module_db_path(PREDICTIVE_LEARNING_DB_PATH))
        except Exception as exc:
            logger.warning("Predictive learning module unavailable: %s", exc)
            return None

    def _create_echowiring_memory(self):
        """Attach the asynchronous/EchoWiring memory module when present.

        Supports both:
        - Asyncronous Learning.py  (preferred)
        - Asyncronous Learning.txt (legacy)
        """
        module = None
        for file_name in (ASYNCHRONOUS_LEARNING_MODULE_FILE, ASYNCHRONOUS_LEARNING_MODULE_FALLBACK_FILE):
            module = (
                self._get_loaded_module_by_filename(file_name)
                or self._load_text_module("perseus_asynchronous_learning", file_name)
            )
            if module and getattr(module, "EchoWiringMemory", None):
                break

        cls = getattr(module, "EchoWiringMemory", None) if module else None
        if not cls:
            logger.warning(
                "EchoWiring memory module unavailable: expected %s or %s in %s",
                ASYNCHRONOUS_LEARNING_MODULE_FILE,
                ASYNCHRONOUS_LEARNING_MODULE_FALLBACK_FILE,
                self._modules_root(),
            )
            return None
        try:
            return cls(db_path=self._module_db_path(ECHOWIRING_MEMORY_DB_PATH))
        except Exception as exc:
            logger.warning("EchoWiring memory module unavailable: %s", exc)
            return None

    def _create_cognitive_engine(self):
        """Attach the Cognitive Functions module when present."""
        module = self._get_loaded_module_by_filename(COGNITIVE_FUNCTIONS_MODULE_FILE) or self._load_text_module("perseus_cognitive_functions", COGNITIVE_FUNCTIONS_MODULE_FILE)
        cls = getattr(module, "GhostCoreCognitiveEngine", None) if module else None
        db_cls = getattr(module, "CognitiveMemoryDB", None) if module else None
        if not cls:
            return None
        try:
            engine = cls()
            if db_cls:
                engine.db = db_cls(db_path=self._module_db_path(COGNITIVE_STATE_DB_PATH))
            return engine
        except Exception as exc:
            logger.warning("Cognitive functions module unavailable: %s", exc)
            return None

    def _create_brain_state_engine(self):
        """Attach the deterministic Brain State module when present."""
        module = self._get_loaded_module_by_filename(BRAIN_STATE_MODULE_FILE) or self._load_text_module("perseus_brain_state", BRAIN_STATE_MODULE_FILE)
        cls = getattr(module, "BrainStateEngine", None) if module else None
        if not cls:
            return None

        try:
            return cls(db_path=self._module_db_path(BRAIN_STATE_DB_PATH))
        except Exception as exc:
            logger.warning("Brain State module unavailable: %s", exc)
            return None

    def _create_search_augmentation(self):
        """Attach online search augmentation; fall back to the built-in lightweight searcher if the module is absent."""
        if not self.allow_online_search:
            return None

        module = self._get_loaded_module_by_filename(SEARCH_AUGMENTATION_MODULE_FILE) or self._load_text_module("perseus_search_augmentation", SEARCH_AUGMENTATION_MODULE_FILE)
        cls = getattr(module, "SearchAugmentation", None) if module else None
        if cls:
            try:
                return cls(db_path=self._module_db_path(SEARCH_CACHE_DB_PATH), allow_network=self.allow_online_search)
            except Exception as exc:
                logger.warning("Search augmentation module unavailable, using built-in fallback: %s", exc)

        try:
            return BasicSearchAugmentation(db_path=self._module_db_path(SEARCH_CACHE_DB_PATH), allow_network=self.allow_online_search)
        except Exception as exc:
            logger.warning("Built-in search augmentation unavailable: %s", exc)
            return None

    def _create_english_language_engine(self):
        """Attach the English Language comprehension module when present."""
        module = self._get_loaded_module_by_filename(ENGLISH_LANGUAGE_MODULE_FILE) or self._load_text_module("perseus_english_language", ENGLISH_LANGUAGE_MODULE_FILE)
        cls = getattr(module, "EnglishLanguageModule", None) if module else None
        if not cls:
            return None

        try:
            return cls()
        except Exception as exc:
            logger.warning("English Language module unavailable: %s", exc)
            return None

    def _create_autonomous_trainer(self):
        """Attach the autonomous training-data module when present."""
        module = self._get_loaded_module_by_filename(AUTONOMOUS_TRAINING_MODULE_FILE) or self._load_text_module("perseus_autonomous_training", AUTONOMOUS_TRAINING_MODULE_FILE)
        cls = getattr(module, "AutonomousTrainingMemory", None) if module else None
        if not cls:
            return None

        try:
            return cls(
                db_path=self._module_db_path(AUTONOMOUS_TRAINING_DB_PATH),
                dataset_dir=str(Path(__file__).resolve().parent / "training_datasets"),
                min_quality_score=self._quality_threshold,
            )
        except Exception as exc:
            logger.warning("Autonomous training module unavailable: %s", exc)
            return None

    def _load_adjacent_python_text_module(self, module_name: str, file_name: str):
        """
        Load a Python-compatible module from the same folder as portable_llm.py.

        This complements _load_text_module(), which looks inside Modules/. It lets
        development copies like "Introspective Learning.py" sit beside portable_llm.py
        without requiring a Modules/ move first.
        """
        path = Path(__file__).resolve().parent / file_name
        if not path.exists():
            return None

        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            module.__package__ = ""
            sys.modules[module_name] = module
            exec(compile(source, str(path), "exec"), module.__dict__)
            return module
        except Exception as exc:
            sys.modules.pop(module_name, None)
            logger.warning("Unable to load adjacent module %s from %s: %s", module_name, path, exc)
            return None

    def _create_introspective_learning(self):
        """Attach the post-response repair / self-review layer when present."""
        module = (
            self._get_loaded_module_by_filename(INTROSPECTIVE_LEARNING_MODULE_FILE)
            or self._load_text_module("perseus_introspective_learning", INTROSPECTIVE_LEARNING_MODULE_FILE)
            or self._load_adjacent_python_text_module("perseus_introspective_learning_adjacent", INTROSPECTIVE_LEARNING_MODULE_FILE)
        )
        cls = getattr(module, "IntrospectiveLearning", None) if module else None
        if not cls:
            return None

        try:
            return cls(db_path=self._module_db_path(INTROSPECTIVE_LEARNING_DB_PATH))
        except Exception as exc:
            logger.warning("Introspective Learning module unavailable: %s", exc)
            return None

    @staticmethod
    def _quality_from_introspective_critique(critique, fallback_score: int = 70) -> ResponseQuality:
        """Convert an IntrospectiveLearning critique object into PortableLLM's quality object."""
        try:
            directness = int(getattr(critique, "directness_score", fallback_score))
            relevance = int(getattr(critique, "relevance_score", fallback_score))
            completeness = int(getattr(critique, "completeness_score", fallback_score))

            score = round((directness * 0.35) + (relevance * 0.35) + (completeness * 0.30))
            reasons = list(getattr(critique, "issues", []) or [])

            if getattr(critique, "leakage_detected", False):
                score -= 35
                if "Internal leakage detected." not in reasons:
                    reasons.append("Internal leakage detected.")
            if getattr(critique, "harmful_scaffolding_detected", False):
                score -= 25
                if "Harmful scaffolding detected." not in reasons:
                    reasons.append("Harmful scaffolding detected.")
            if not getattr(critique, "answered_question", True):
                score -= 20
                if "Did not clearly answer the user's question." not in reasons:
                    reasons.append("Did not clearly answer the user's question.")

            score = max(0, min(100, int(score)))
            if not reasons:
                reasons = ["Introspective review passed"]

            return ResponseQuality(score=score, reasons=reasons)
        except Exception:
            return ResponseQuality(score=int(fallback_score), reasons=["Introspective review score fallback used"])

    def _apply_introspective_repair(
        self,
        prompt: str,
        response: Optional[str],
        profile: PromptProfile,
        enriched: EnrichedPrompt,
        current_quality: ResponseQuality,
    ) -> Tuple[Optional[str], ResponseQuality, bool, Optional[Dict[str, object]]]:
        """
        Final self-review gate before the answer is shown or captured as memory/training data.

        The introspection module is allowed to repair weak answers and strip leaked
        scaffolding, but it never gets to expose its own critique to the user.
        """
        engine = getattr(self, "introspective_learning", None)
        if not engine or not response:
            return response, current_quality, False, None

        try:
            repaired, critique = engine.analyze_and_correct(
                user_prompt=prompt,
                response_text=str(response),
                rewrite_callback=None,
                search_context=enriched.context_preview,
            )
            repaired = _sanitize_visible_response(repaired, prompt_payload=enriched.text)
            if not repaired:
                return response, current_quality, False, None

            introspective_quality = self._quality_from_introspective_critique(
                critique,
                fallback_score=current_quality.score,
            )

            # Keep the stricter score when the review found issues. This prevents
            # weak/repaired answers from being over-promoted into training data.
            if repaired.strip() != str(response).strip():
                final_quality = introspective_quality
                changed = True
            elif introspective_quality.score < current_quality.score:
                final_quality = introspective_quality
                changed = False
            else:
                final_quality = current_quality
                if introspective_quality.reasons and introspective_quality.reasons != ["Introspective review passed"]:
                    final_quality = ResponseQuality(
                        score=current_quality.score,
                        reasons=list(dict.fromkeys(list(current_quality.reasons or []) + introspective_quality.reasons)),
                    )
                changed = False

            meta = {
                "score": introspective_quality.score,
                "reasons": introspective_quality.reasons,
                "changed_response": changed,
                "confidence": getattr(critique, "confidence", None),
                "answered_question": getattr(critique, "answered_question", None),
                "leakage_detected": getattr(critique, "leakage_detected", None),
                "harmful_scaffolding_detected": getattr(critique, "harmful_scaffolding_detected", None),
            }
            return repaired, final_quality, changed, meta
        except Exception as exc:
            logger.warning("Introspective Learning repair failed: %s", exc)
            return response, current_quality, False, {"error": str(exc)}

    def _enrich_prompt_with_language_engine(self, prompt: str) -> EnrichedPrompt:
        """
        Run a pre-analysis English comprehension pass before knowledge/context analysis.

        This module is intentionally hidden from user output. It tells the model how to
        understand the user's wording, implied task, ambiguity, tone, and expected answer
        shape before it starts reasoning over retrieved context.
        """
        if _is_small_talk_prompt(prompt) or _is_capability_prompt(prompt):
            return EnrichedPrompt(text=prompt, has_context=False)

        engine = getattr(self, "english_language_engine", None)
        if not engine:
            return EnrichedPrompt(text=prompt, has_context=False)

        try:
            if hasattr(engine, "build_prompt_context"):
                language_context = engine.build_prompt_context(prompt)
            elif hasattr(engine, "analyze"):
                analysis = engine.analyze(prompt)
                language_context = json.dumps(analysis, ensure_ascii=False, indent=2)
            else:
                return EnrichedPrompt(text=prompt, has_context=False)
        except Exception as exc:
            logger.warning("English Language pre-analysis failed: %s", exc)
            return EnrichedPrompt(text=prompt, has_context=False)

        language_context = (language_context or "").strip()
        if not language_context:
            return EnrichedPrompt(text=prompt, has_context=False)

        preview = language_context[:1200].replace("\n", " ").strip()
        enriched_text = (
            "Before answering, use this English-language comprehension analysis as hidden guidance. "
            "It is not user-visible content. Do not quote it, label it, or dump it back. "
            "Use it to understand the request, disambiguate wording, choose the right answer shape, "
            "and identify what must be answered directly before using any other context.\n\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_BEGIN\n"
            "English language pre-analysis:\n"
            f"{language_context}\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_END\n\n"
            "User request:\n"
            f"{prompt}"
        )
        return EnrichedPrompt(text=enriched_text, has_context=True, context_preview=preview)

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
        brain_meta = None
        if getattr(self, "brain_state_engine", None):
            try:
                profile_packet = {
                    "intent": profile.intent,
                    "complexity": profile.complexity,
                    "mood": profile.mood,
                    "expected_shape": profile.expected_shape,
                }
                brain_state, brain_action = self.brain_state_engine.step_input(prompt, profile=profile_packet)
                self._active_brain_action = brain_action
                self._active_brain_context = self.brain_state_engine.build_llm_context(brain_action)
                brain_meta = {
                    "intent": getattr(brain_action, "intent", None),
                    "goal": getattr(brain_action, "goal", None),
                    "strategy": getattr(brain_action, "response_strategy", None),
                    "retrieval_strategy": getattr(brain_action, "retrieval_strategy", None),
                    "confidence": getattr(brain_action, "confidence", None),
                    "focus_terms": list(getattr(brain_action, "focus_terms", []) or [])[:8],
                    "update_count": getattr(brain_state, "update_count", None),
                }
            except Exception as exc:
                logger.warning("Brain State input update failed: %s", exc)
                self._active_brain_action = None
                self._active_brain_context = ""

        enriched = self._enrich_prompt_with_language_engine(prompt)
        enriched = self._enrich_prompt_with_dynamic_modules(enriched, prompt)
        enriched = self._merge_enriched_prompts(enriched, self._enrich_prompt_with_knowledge(prompt))
        enriched = self._enrich_prompt_with_online_search(enriched, prompt)
        enriched = self._enrich_prompt_with_predictive_modules(enriched, prompt)
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
            response = _sanitize_visible_response(response, prompt_payload=prompt)
            provider_used = "offline"
            quality = self._assess_quality(response, profile)

        if enriched.has_context and response and quality.score < self._quality_threshold:
            response = self._build_grounded_response(prompt=prompt, context=enriched.context_preview)
            response = _sanitize_visible_response(response, prompt_payload=enriched.text)
            provider_used = "grounded-fallback"
            quality = self._assess_quality(response, profile, has_context=True)
            refined = True

        sanitized_response = _sanitize_visible_response(response, prompt_payload=enriched.text)
        if response and sanitized_response != str(response).strip():
            response = sanitized_response
            quality = self._assess_quality(response, profile, has_context=enriched.has_context)
            refined = True

        introspection_meta = None
        response, quality, introspection_changed, introspection_meta = self._apply_introspective_repair(
            prompt=prompt,
            response=response,
            profile=profile,
            enriched=enriched,
            current_quality=quality,
        )
        if introspection_changed:
            refined = True

        # Ollama is the rare rescue path: only use it after the local/fallback answer,
        # grounding rescue, sanitizer, and introspection have all failed to clear the
        # quality threshold.
        if (
            response
            and quality.score < self._quality_threshold
            and provider_used != "ollama"
            and "ollama" in self.manager.providers
            and getattr(self.manager.providers.get("ollama"), "available", False)
        ):
            ollama_response = self._generate_with_provider("ollama", enriched.text, profile, refine=False, prior_response=None)
            ollama_quality = self._assess_quality(ollama_response, profile, has_context=enriched.has_context)
            ollama_response, ollama_quality, ollama_changed, ollama_intro_meta = self._apply_introspective_repair(
                prompt=prompt,
                response=ollama_response,
                profile=profile,
                enriched=enriched,
                current_quality=ollama_quality,
            )
            if ollama_response and ollama_quality.score > quality.score:
                response = ollama_response
                quality = ollama_quality
                provider_used = "ollama"
                refined = bool(refined or ollama_changed)
                introspection_meta = ollama_intro_meta or introspection_meta

        if response and str(response).strip() and getattr(self, "brain_state_engine", None):
            try:
                self.brain_state_engine.update_after_response(
                    input_text=prompt,
                    response_text=str(response),
                    quality_score=int(quality.score),
                    issues=list(quality.reasons or [])[:4],
                )
            except Exception as exc:
                logger.warning("Brain State post-response update failed: %s", exc)

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
                    "brain_state": brain_meta,
                    "introspection": introspection_meta,
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
            self._learn_predictive_modules_from_turn(
                prompt=prompt,
                response=str(response),
                profile=profile,
                provider=provider_used,
                quality=quality,
            )
            self._learn_autonomous_training_from_turn(
                prompt=prompt,
                response=str(response),
                profile=profile,
                provider=provider_used,
                model=model_used,
                quality=quality,
                context_preview=enriched.context_preview,
                refined=refined,
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
                "brain_state": brain_meta,
                "introspection": introspection_meta,
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
        context_bullets = _context_preview_bullets(context)
        bullet_text = "\n".join(f"- {item}" for item in context_bullets)
        return (
            "Summary:\n"
            "I found relevant learned context for this request. The safe answer is to use that context as the base, "
            "separate confirmed details from assumptions, and turn the result into concrete next actions instead of guessing.\n\n"
            "Ingested Context Used:\n"
            f"{bullet_text or '- Relevant learned context was retrieved, but the preview was too small to summarize deterministically.'}\n\n"
            "Reasoned Takeaway:\n"
            "- What matters: answer from the retrieved facts first, then add interpretation only where the evidence supports it.\n"
            "- Why it matters: this keeps Perseus useful without pretending a weak model response was stronger than it was.\n"
            "- Uncertainty: if the retrieved context is incomplete, validate against primary docs, local files, or source material before relying on it.\n\n"
            "Next Steps:\n"
            "1. Ask a narrower follow-up naming the file, source, feature, or decision you want analyzed.\n"
            "2. Add or ingest the missing source material if the context above is thin.\n"
            "3. Use the model-backed path with Ollama running for a fuller synthesized answer.\n\n"
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
            "predictive_learning_enabled": bool(self.predictive_memory),
            "echowiring_memory_enabled": bool(self.echowiring_memory),
            "cognitive_functions_enabled": bool(self.cognitive_engine),
            "online_search_enabled": bool(self.search_augmentation),
            "english_language_module_enabled": bool(getattr(self, "english_language_engine", None)),
            "autonomous_training_enabled": bool(getattr(self, "autonomous_trainer", None)),
            "introspective_learning_enabled": bool(getattr(self, "introspective_learning", None)),
            "autonomous_training": self._autonomous_training_stats(),
            "loaded_modules_count": sum(1 for item in getattr(self, "module_load_report", []) if item.get("loaded")),
            "dynamic_module_engines_count": len(getattr(self, "dynamic_module_engines", {}) or {}),
        }

    def _autonomous_training_stats(self) -> Dict[str, object]:
        trainer = getattr(self, "autonomous_trainer", None)
        if not trainer or not hasattr(trainer, "get_stats"):
            return {}
        try:
            return trainer.get_stats()
        except Exception as exc:
            logger.warning("Autonomous training stats failed: %s", exc)
            return {"error": str(exc)}

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

    def export_brain_state(self) -> Dict[str, object]:
        """Return the current deterministic brain-state snapshot."""
        if not getattr(self, "brain_state_engine", None):
            return {"ok": False, "error": "Brain State module not loaded"}
        try:
            return {"ok": True, "state": self.brain_state_engine.export_state()}
        except Exception as exc:
            logger.warning("Brain State export failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def export_training_dataset(self, output_path: str = "", format: str = "chatml", limit: int = 5000) -> Dict[str, object]:
        """Export accepted autonomous-training examples for a future native Perseus model."""
        trainer = getattr(self, "autonomous_trainer", None)
        if not trainer or not hasattr(trainer, "export_dataset"):
            return {"ok": False, "error": "Autonomous Training module not loaded"}
        try:
            return trainer.export_dataset(output_path=output_path, format=format, accepted_only=True, limit=limit)
        except Exception as exc:
            logger.warning("Autonomous training dataset export failed: %s", exc)
            return {"ok": False, "error": str(exc)}

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

        memory_summary = self._extract_memory_summary(
            prompt=prompt_text,
            response=response_text,
            profile=profile,
            quality=quality,
        )
        if not memory_summary and quality.score < self._quality_threshold and len(prompt_text.split()) <= 4:
            return

        memory_block = "\n".join(f"- {item}" for item in memory_summary)
        memory_categories = self._memory_categories(memory_summary)
        retrieval_terms = " ".join(self._memory_retrieval_keywords(prompt_text, memory_summary, profile))

        content = (
            "Learned chat interaction:\n"
            f"Timestamp: {datetime.utcnow().isoformat(timespec='seconds')}Z\n"
            f"Intent: {profile.intent}\n"
            f"Complexity: {profile.complexity}\n"
            f"Provider: {provider}\n"
            f"Quality score: {quality.score}\n\n"
            "Memory categories:\n"
            f"{', '.join(memory_categories) or 'topic'}\n\n"
            "Memory summary:\n"
            f"{memory_block or '- User asked about this topic; preserve the topic and answer style for future context.'}\n\n"
            "Retrieval keywords:\n"
            f"{retrieval_terms}\n\n"
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

    def _learn_predictive_modules_from_turn(
        self,
        prompt: str,
        response: str,
        profile: PromptProfile,
        provider: str,
        quality: ResponseQuality,
    ) -> None:
        """Write the completed turn into the three Modules/ learning engines."""
        prompt_text = re.sub(r"\s+", " ", (prompt or "").strip())
        response_text = re.sub(r"\s+", " ", (response or "").strip())
        if not prompt_text or not response_text:
            return

        memory_summary = self._extract_memory_summary(prompt_text, response_text, profile, quality)
        lesson = _predictive_lesson_for_turn(profile, quality, memory_summary)
        tags = [
            "perseus",
            "chat_turn",
            f"intent:{profile.intent}",
            f"quality:{quality.score}",
            f"provider:{provider}",
        ]
        tags.extend(self._memory_categories(memory_summary))
        context = (
            f"Intent={profile.intent}; complexity={profile.complexity}; provider={provider}; "
            f"quality={quality.score}; reasons={'; '.join(quality.reasons[:6])}; "
            f"memory_summary={' | '.join(memory_summary[:MAX_MEMORY_SUMMARY_BULLETS])}"
        )[:1200]

        if self.predictive_memory:
            try:
                self.predictive_memory.add_event(
                    who="user",
                    what=prompt_text[:900],
                    why=f"User requested a {profile.intent} response; Perseus should learn the pattern and outcome.",
                    context=context,
                    outcome=response_text[:1200],
                    lesson=lesson,
                    confidence=max(0.1, min(0.98, quality.score / 100)),
                    tags=tags,
                )
            except Exception as exc:
                logger.warning("Predictive learning event write failed: %s", exc)

        if self.echowiring_memory:
            try:
                self.echowiring_memory.add_event(
                    who="user",
                    what=prompt_text[:900],
                    why=f"User interaction created a reusable lesson for future response prediction ({profile.intent}).",
                    context=context,
                    outcome=response_text[:1200],
                    lesson=lesson,
                    confidence=max(0.1, min(0.98, quality.score / 100)),
                    tags=tags + ["echowiring_memory"],
                    amm_enabled=False,
                    consent_confirmed=False,
                    safety_notes="No AMM cues were recorded; this is ordinary predictive chat memory only.",
                )
            except Exception as exc:
                logger.warning("EchoWiring learning event write failed: %s", exc)

        if self.cognitive_engine:
            try:
                traces = self.cognitive_engine.retrieve_relevant_memory(prompt_text)
                analysis = self.cognitive_engine.deconstruct_response(prompt_text, response_text)
                reevaluation = self.cognitive_engine.reevaluate_response(prompt_text, response_text, analysis, traces)
                updated_state = self.cognitive_engine.memory_update(
                    brain_state=self.cognitive_engine.brain_state,
                    input_text=prompt_text,
                    response_text=response_text,
                    analysis=analysis,
                    reevaluation=reevaluation,
                )
                self.cognitive_engine.brain_state = updated_state
                self.cognitive_engine.db.save_brain_snapshot(updated_state)
            except Exception as exc:
                logger.warning("Cognitive learning update failed: %s", exc)

    def _learn_autonomous_training_from_turn(
        self,
        prompt: str,
        response: str,
        profile: PromptProfile,
        provider: str,
        model: str,
        quality: ResponseQuality,
        context_preview: str = "",
        refined: bool = False,
    ) -> None:
        """Store clean high-quality turns as candidate deep-learning examples."""
        trainer = getattr(self, "autonomous_trainer", None)
        if not trainer or not hasattr(trainer, "add_interaction"):
            return

        try:
            trainer.add_interaction(
                prompt=prompt,
                response=response,
                intent=profile.intent,
                provider=provider,
                model=model,
                quality_score=quality.score,
                quality_reasons=list(quality.reasons or []),
                context_preview=context_preview,
                metadata={
                    "complexity": profile.complexity,
                    "mood": profile.mood,
                    "expected_shape": profile.expected_shape,
                    "refined": bool(refined),
                    "strict_local_only": self.strict_local_only,
                },
            )
        except Exception as exc:
            logger.warning("Autonomous training capture failed: %s", exc)

    @staticmethod
    def _memory_categories(memories: List[str]) -> List[str]:
        categories: List[str] = []
        mapping = [
            ("preference", ["preference", "values", "expects", "prefers"]),
            ("project", ["project", "repo", "assistant", "tool", "working on"]),
            ("identity", ["identity", "name", "self-context", "call"]),
            ("correction", ["correction", "not ", "instead", "avoid"]),
            ("style", ["style", "tone", "sophisticated", "intelligent", "candid"]),
            ("task", ["topic", "task", "roadmap", "plan", "analysis"]),
        ]
        blob = "\n".join(memories).lower()
        for category, markers in mapping:
            if any(marker in blob for marker in markers):
                categories.append(category)
        return categories or (["topic"] if memories else [])

    @staticmethod
    def _memory_retrieval_keywords(prompt: str, memories: List[str], profile: PromptProfile) -> List[str]:
        """Build dense, low-noise terms that make chat memories easy to retrieve later."""
        seed_text = f"{prompt} {profile.intent} {' '.join(memories)}"
        terms = [
            term.lower()
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", seed_text)
            if term.lower() not in MEMORY_RETRIEVAL_STOPWORDS
        ]
        phrases = re.findall(
            r"(?:called|named|project is|repo is|prefer|want|need)\s+([^.!?;]{2,80})",
            prompt,
            flags=re.IGNORECASE,
        )
        phrase_terms = [re.sub(r"\s+", " ", phrase).strip(" ,:-").lower() for phrase in phrases]

        ordered: List[str] = ["chat-memory", "memory-summary", f"intent-{profile.intent}"]
        if any("preference" in item.lower() or "values" in item.lower() for item in memories):
            ordered.extend(["user-preference", "response-style"])
        if any("project" in item.lower() for item in memories):
            ordered.extend(["project-context", "user-project"])
        if any("correction" in item.lower() or "avoid" in item.lower() for item in memories):
            ordered.extend(["correction", "avoid-repeat"])

        for item in phrase_terms + terms:
            if item and item not in ordered:
                ordered.append(item)
            if len(ordered) >= 45:
                break
        return ordered

    @staticmethod
    def _extract_memory_summary(
        prompt: str,
        response: str,
        profile: PromptProfile,
        quality: ResponseQuality,
    ) -> List[str]:
        """Extract compact durable memories from a chat turn without needing another model."""
        text = re.sub(r"\s+", " ", (prompt or "").strip())
        lower = text.lower()
        memories: List[str] = []

        def add_memory(item: str) -> None:
            item = re.sub(r"\s+", " ", item).strip(" .")
            if not item:
                return
            if _looks_sensitive_memory(item):
                return
            if _looks_low_value_memory(item):
                return
            if item.lower() in {m.lower() for m in memories}:
                return
            memories.append(item[:260])

        explicit_patterns = [
            (r"\bremember (?:that )?(.+?)(?:[.!?]|$)", "User explicitly asked Perseus to remember: {}"),
            (r"\b(?:call me|my name is)\s+(.+?)(?:[.!?]|$)", "User identity/name preference: {}"),
            (r"\b(?:i prefer|i like|i want|i need)\s+(.+?)(?:[.!?]|$)", "User preference: {}"),
            (r"\b(?:always|please always)\s+(.+?)(?:[.!?]|$)", "User standing instruction: always {}"),
            (r"\b(?:never|do not|don't)\s+(.+?)(?:[.!?]|$)", "User standing instruction: avoid {}"),
            (r"\b(?:correct(?:ion)?|actually)[:,]?\s+(.+?)(?:[.!?]|$)", "User correction: {}"),
            (r"\bmy (?:project|app|repo|repository|tool|assistant) (?:is|called|named)\s+(.+?)(?:[.!?]|$)", "User project context: {}"),
            (r"\b(?:we are|we're|i am|i'm|im) (?:building|making|working on|creating)\s+(.+?)(?:[.!?]|$)", "User current project/work: {}"),
            (r"\b(?:i am|i'm|im)\s+(.+?)(?:[.!?]|$)", "User self-context: {}"),
        ]
        for pattern, template in explicit_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(1).strip(" ,;:-")
                if 2 <= len(value) <= 180:
                    add_memory(template.format(value))

        if any(marker in lower for marker in ["sophisticated", "intelligent", "smart", "educated response"]):
            add_memory("User values sophisticated, intelligent, educated responses over generic assistant filler")
        if any(marker in lower for marker in ["learn as", "learn from", "remember", "memory"]):
            add_memory("User expects Perseus to learn from useful chat turns and reuse that memory later")
        if any(marker in lower for marker in ["local", "no other ai", "without other ai", "wouldnt need to use any other ais", "wouldn't need to use any other ais"]):
            add_memory("User prefers Perseus to rely on local learning and local models instead of other AI services")

        topic_terms = [
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", lower)
            if term not in {
                "that",
                "this",
                "with",
                "from",
                "have",
                "want",
                "need",
                "using",
                "would",
                "could",
                "should",
                "response",
                "responses",
            }
        ]
        if topic_terms:
            add_memory(f"Recent user topic keywords: {', '.join(dict.fromkeys(topic_terms[:10]))}")

        if response and quality.score >= 80:
            response_terms = [
                term
                for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{4,}", response.lower())
                if term not in {"because", "therefore", "which", "their", "there", "about", "would", "should"}
            ]
            if response_terms:
                add_memory(f"High-quality answer touched: {', '.join(dict.fromkeys(response_terms[:8]))}")

        if profile.intent in {"feedback", "strategic", "analytical"}:
            add_memory(f"For {profile.intent} prompts, user benefits from candid judgment, trade-offs, and concrete next steps")

        correction_match = re.search(r"\bnot\s+(.{2,80}?)\s+but\s+(.{2,100}?)(?:[.!?]|$)", text, flags=re.IGNORECASE)
        if correction_match:
            add_memory(
                "User correction: not "
                f"{correction_match.group(1).strip(' ,;:-')} but {correction_match.group(2).strip(' ,;:-')}"
            )

        if any(marker in lower for marker in ["too generic", "generic", "not specific", "more specific", "vague"]):
            add_memory("User dislikes generic or vague answers; increase specificity, concrete examples, and direct judgment")
        if any(marker in lower for marker in ["heuristic", "heuristics", "rules", "signals"]):
            add_memory("User values explicit heuristics and decision rules that improve future answer quality")

        return memories[:MAX_MEMORY_SUMMARY_BULLETS]

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
        folder_label = _knowledge_folder_label(root)

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

            title = f"{folder_label}/{relative_title}"

            ingest = self.ingest_web_content(url=path.resolve().as_uri(), content=content, title=title)
            ingest["path"] = str(path)
            ingest["title"] = title
            results.append(ingest)
            if ingest.get("ok"):
                successes += 1
                learned_titles.append(title)

        if successes > 0:
            index_content = _build_folder_index_content(root=root, files=files, learned_titles=learned_titles)
            index_title = f"{folder_label}/_folder_index"
            index_ingest = self.ingest_web_content(
                url=f"perseus://folder-index/{_safe_knowledge_id(folder_label)}",
                content=index_content,
                title=index_title,
            )
            index_ingest["path"] = str(root)
            index_ingest["title"] = index_title
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

    @staticmethod
    def _merge_enriched_prompts(primary: EnrichedPrompt, secondary: EnrichedPrompt) -> EnrichedPrompt:
        """Merge hidden context enrichments without losing the original user request."""
        if not primary.has_context:
            return secondary
        if not secondary.has_context:
            return primary

        merged_text = (
            f"{primary.text}\n\n"
            "Additional hidden retrieved context follows. Use it as evidence only; do not paste it raw.\n\n"
            f"{secondary.text}"
        )
        preview_parts = [primary.context_preview, secondary.context_preview]
        preview = " | ".join(part for part in preview_parts if part)[:1200]
        return EnrichedPrompt(text=merged_text, has_context=True, context_preview=preview)

    def _enrich_prompt_with_knowledge(self, prompt: str) -> EnrichedPrompt:
        """Inject relevant learned web context into the prompt when available."""
        if _is_small_talk_prompt(prompt) or _is_capability_prompt(prompt):
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

        preview = context[:1200].replace("\n", " ").strip()
        enriched_text = (
            "You have retrieved ingested web knowledge relevant to this query. "
            "Ground your answer in that ingested context, including learned user chat memory when relevant, "
            "and be explicit when you do. Use the context as internal evidence; do not paste raw retrieved excerpts.\n\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_BEGIN\n"
            "Ingested context:\n"
            f"{context}\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_END\n\n"
            "Output requirements:\n"
            "1. Start with a realistic summary of what is currently known.\n"
            "2. Review all retrieved ingested context blocks, synthesize every relevant non-duplicate point, "
            "and include an 'Ingested Context Used' section with paraphrased concrete points, not raw context dumps.\n"
            "3. Include uncertainty where evidence is incomplete.\n"
            "4. Use learned user preferences, prior chat facts, and project context when they are relevant.\n"
            "5. Provide educational explanation and practical implications.\n"
            "6. End with a 'Next Steps' section focused on action.\n\n"
            "User request:\n"
            f"{prompt}"
        )

        return EnrichedPrompt(text=enriched_text, has_context=True, context_preview=preview)

    def _enrich_prompt_with_predictive_modules(self, enriched: EnrichedPrompt, prompt: str) -> EnrichedPrompt:
        """Inject context from the three Modules/ learning engines when available."""
        module_context = self._build_predictive_module_context(prompt)
        if not module_context:
            return enriched

        enriched_text = (
            "You have additional predictive/cognitive learning context from local Perseus modules. "
            "Use it as probabilistic background, not as certainty. Prefer explicit user corrections, standing instructions, "
            "prior lessons, and cognitive risk notes when relevant.\n\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_BEGIN\n"
            f"{module_context}\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_END\n\n"
            "Current prompt payload:\n"
            f"{enriched.text}"
        )
        preview_parts = [part for part in [enriched.context_preview, module_context[:800].replace("\n", " ").strip()] if part]
        return EnrichedPrompt(text=enriched_text, has_context=True, context_preview=" | ".join(preview_parts))

    def _enrich_prompt_with_online_search(self, enriched: EnrichedPrompt, prompt: str) -> EnrichedPrompt:
        """Inject current online search context when local knowledge is missing or the prompt needs freshness."""
        searcher = getattr(self, "search_augmentation", None)
        if not searcher:
            return enriched
        if _is_small_talk_prompt(prompt) or _is_capability_prompt(prompt):
            return enriched

        try:
            decision = searcher.should_search(prompt, local_context=enriched.context_preview)
        except Exception as exc:
            logger.warning("Search decision failed: %s", exc)
            return enriched

        if not getattr(decision, "should_search", False):
            return enriched

        try:
            search_context = searcher.search_and_build_context(prompt)
        except Exception as exc:
            logger.warning("Online search failed: %s", exc)
            return enriched

        if not search_context:
            return enriched

        enriched_text = (
            "You have online search context for a current or unknown-information request. "
            "Use it as internal evidence to analyze the request, not as text to display. "
            "Do not expose internal predictive/cognitive scaffolding, lookup payloads, context blocks, or raw snippets.\n\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_BEGIN\n"
            f"Search decision: {getattr(decision, 'reason', 'Online lookup requested')}\n\n"
            f"{search_context}\n"
            "RAW_CONTEXT_DO_NOT_OUTPUT_END\n\n"
            "Output requirements:\n"
            "1. Answer the user's request directly from the search context when possible.\n"
            "2. Synthesize and paraphrase; do not quote or paste source snippets/context verbatim.\n"
            "3. Include a short 'Sources consulted' note with domains or provider names only when useful.\n"
            "4. If the snippets are thin, say what is uncertain and suggest a narrower location/source check.\n"
            "5. Do not tell the user to ingest sources unless online search returned no usable information.\n\n"
            "User request:\n"
            f"{prompt}"
        )
        preview_parts = [
            part
            for part in [
                enriched.context_preview,
                search_context[:1200].replace("\n", " ").strip(),
            ]
            if part
        ]
        return EnrichedPrompt(text=enriched_text, has_context=True, context_preview=" | ".join(preview_parts))

    def _build_predictive_module_context(self, prompt: str) -> str:
        """Read predictions/lessons from Predictive, EchoWiring, and Cognitive modules."""
        blocks: List[str] = []

        if self.predictive_memory:
            try:
                packet = self.predictive_memory.predict_from_context(prompt, limit=5)
                blocks.append(_format_prediction_packet("PREDICTIVE LEARNING", packet))
            except Exception as exc:
                logger.warning("Predictive learning context failed: %s", exc)

        if self.echowiring_memory:
            try:
                packet = self.echowiring_memory.predict_from_context(prompt, limit=5)
                blocks.append(_format_prediction_packet("ASYNCHRONOUS / ECHOWIRING LEARNING", packet))
            except Exception as exc:
                logger.warning("EchoWiring context failed: %s", exc)

        if self.cognitive_engine:
            try:
                traces = self.cognitive_engine.retrieve_relevant_memory(prompt)
                cognitive_block = _format_cognitive_traces(traces)
                if cognitive_block:
                    blocks.append(cognitive_block)
            except Exception as exc:
                logger.warning("Cognitive context failed: %s", exc)

        filtered = [block for block in blocks if block and "No strong" not in block]
        return "\n\n".join(filtered).strip()[:5000]

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

        if _is_memory_prompt(prompt):
            queries.extend(
                [
                    "chat-memory memory-summary",
                    "Memory summary User preference",
                    "User project context User standing instruction",
                    "User values sophisticated intelligent responses",
                    "User expects Perseus to learn from useful chat turns",
                ]
            )

        cve_matches = re.findall(r"cve-\d{4}-\d{4,}", lower, flags=re.IGNORECASE)
        cwe_matches = re.findall(r"cwe-\d+", lower, flags=re.IGNORECASE)
        for m in cve_matches + cwe_matches:
            queries.append(m.upper())

        quoted_phrases = re.findall(r"['\"]([^'\"]{3,80})['\"]", prompt)
        queries.extend(phrase.strip() for phrase in quoted_phrases if phrase.strip())

        tokens = re.findall(r"[a-zA-Z]{3,}", lower)
        stop = MEMORY_RETRIEVAL_STOPWORDS | {
            "content",
            "explain",
            "folder",
            "ingested",
            "learned",
            "module",
            "modules",
            "project",
            "summarize",
        }
        keywords = [t for t in tokens if t not in stop]

        if len(keywords) >= 2:
            queries.append(" ".join(keywords[: min(6, len(keywords))]))
        if len(keywords) >= 3:
            queries.append(" ".join(keywords[:3]))
            queries.append(" ".join(keywords[-3:]))
        queries.extend(keywords[:6])

        queries.append(prompt)
        deduped: List[str] = []
        seen = set()
        for query in queries:
            normalized = re.sub(r"\s+", " ", str(query or "")).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

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
        elif any(
            token in lower
            for token in [
                "teach",
                "explain",
                "what is",
                "what are",
                "who ",
                "when ",
                "where ",
                "how does",
                "how do",
                "how can",
                "why",
            ]
        ):
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
        """
        Rank providers for this request.

        In strict-local mode, Ollama is deliberately last. The normal path is the
        deterministic/local fallback plus memory/repair layers; Ollama is reserved
        for the later rescue pass in ask_with_metadata().
        """
        if self.strict_local_only:
            ordered = ["fallback"]
            if self.provider not in {"ollama", "fallback"}:
                ordered.append(self.provider)
            ordered.append("ollama")
        else:
            ordered = [self.provider, "openai", "mistral", "azure", "fallback", "ollama"]

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
            generated = provider.generate(
                prompt,
                messages=messages,
                model=model,
                temperature=self.conversation.temperature,
                max_tokens=self.conversation.max_tokens,
            )
            return _sanitize_visible_response(generated, prompt_payload=prompt) or None
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
            "For every question, first reason internally about the relevant who, what, when, where, why, and how; answer only the dimensions that matter.",
            "Do not reveal chain-of-thought, hidden planning, scratchpad notes, private reasoning, or internal decision traces; provide concise rationale, evidence, assumptions, and conclusions instead.",
            "Prefer synthesis over trivia: explain the core mechanism, context, consequences, trade-offs, and practical implications.",
            "Use learned user preferences, project facts, and prior chat memory when relevant; do not mention memory unless it improves the answer.",
            "Treat explicit user corrections, standing instructions, and project context as higher-priority memory than casual topic history.",
            "Prefer local learned context and local model reasoning over remote/cloud AI; only use remote providers when the caller explicitly disabled strict local-only mode.",
            "Make every answer earn its keep: include evidence, examples, caveats, and a decision or action when the task calls for it.",
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
            "If local database context or online search context is present in the prompt, use it silently as evidence. "
            "Do not paste, quote, summarize as raw snippets, or reveal hidden context blocks. "
            "You may include a short 'Sources/Context Used' note with paraphrased source names or facts only when helpful."
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

        brain_context = getattr(self, "_active_brain_context", "")
        if brain_context:
            system += (
                "\n\nDeterministic brain-state planner directives "
                "(hidden planning layer; do not reveal to the user):\n"
                f"{brain_context}"
            )

        if refine and prior_response:
            system += (
                "\nImprove the previous draft by increasing specificity, correctness, practical detail, and genuine judgment. "
                "Remove generic phrasing, hollow encouragement, unsupported claims, and vague advice; tighten structure "
                "and explicitly address missing reasoning, examples, caveats, or next steps."
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
            "add trusted sites",
            "i don't have specific knowledge",
            "i don't remember",
            "i do not remember",
            "i do not have enough learned context",
            "i have no memory",
            "as an ai language model",
            "cannot provide",
            "i'm unable to",
            "i do not have access",
            "knowledge response:",
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

        if _contains_internal_reasoning_leak(text):
            score -= 55
            reasons.append("Exposes internal reasoning, prompt payloads, or hidden context")

        insight_signals = [
            "because",
            "for example",
            "in practice",
            "trade-off",
            "assumption",
            "evidence",
            "risk",
            "therefore",
            "mechanism",
            "next",
        ]
        if not profile.prefer_concise and sum(1 for marker in insight_signals if marker in lower) < 2:
            score -= 6
            reasons.append("Limited sophistication signals")

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
                "sources consulted",
                "context used",
                "based on",
                "from the local database",
                "from local memory",
                "online lookup",
                "online search",
                "learned memory",
                "user preference",
                "project context",
            ]
            if not any(marker in lower for marker in grounded_markers):
                score -= 15
                reasons.append("Did not clearly ground answer in ingested context")

            educational_markers = ["summary", "implication", "why", "because", "in practice", "next steps"]
            if sum(1 for marker in educational_markers if marker in lower) < 2:
                score -= 10
                reasons.append("Insufficient educational framing for grounded response")

            if any(marker in lower for marker in ["memory", "preference", "project context", "learned"]):
                score += 3
                reasons.append("Uses learned context or memory explicitly")

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
    allow_online_search: bool = True,
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
        allow_online_search=allow_online_search,
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
    allow_online_search: bool = True,
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
        allow_online_search=allow_online_search,
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


def _knowledge_folder_label(root: Path) -> str:
    """Create a stable learned-source label that keeps important parent folders."""
    parts = [part for part in root.parts if part]
    lowered = [part.lower() for part in parts]
    for marker in ("princess protocol", "snap logic", "snap-master"):
        if marker in lowered:
            start = lowered.index(marker)
            return "/".join(parts[start:])
    return root.name


def _safe_knowledge_id(label: str) -> str:
    """Create a compact identifier for synthetic folder-index URLs."""
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", label or "folder").strip("-._")
    return safe or "folder"


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
    path_parts = [part for part in root.parts if part]
    folder_aliases = sorted(
        {
            root.name,
            root.as_posix(),
            str(root),
            *path_parts[-6:],
            " ".join(path_parts[-4:]),
        }
    )
    alias_list = "\n".join(f"- {alias}" for alias in folder_aliases if alias)
    folder_label = _knowledge_folder_label(root)
    return (
        f"Folder knowledge index for {folder_label}.\n"
        f"This folder is part of the local Perseus learning base and should be considered for relevant answers.\n"
        f"Absolute path: {root}\n\n"
        "Folder aliases / path terms:\n"
        f"{alias_list}\n\n"
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
    marker = "Current prompt payload:"
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


def _answer_from_ingested_context(prompt: str, user_prompt: str) -> str:
    """Create a grounded fallback answer from retrieved learned context without dumping raw context."""
    sources = _parse_ingested_context_sources(prompt)
    if not sources:
        return (
            "I found an ingested-context prompt, but the context block was empty or unreadable. "
            "Re-ingest the folder or ask with more specific file/module names."
        )

    points = _synthesize_ingested_points(sources, user_prompt)
    if not points:
        return (
            "I found ingested sources, but I could not extract a confident answer from the retrieved excerpts. "
            "Ask for a narrower file, function, or module name so I can retrieve a tighter slice."
        )

    used_titles = re.findall(r"`([^`]+)`:", "\n".join(points))
    source_note = ", ".join(used_titles[:4] or [source["title"] for source in sources[:1]])
    return (
        "Answer based on ingested content:\n"
        f"{chr(10).join(f'- {point}' for point in points)}\n\n"
        f"Sources consulted: {source_note}.\n\n"
        "Next steps:\n"
        "- Ask about a specific file/function if you want a deeper code walkthrough.\n"
        "- Re-run folder ingest after changing SNAP files so the learned store stays current."
    )


def _parse_ingested_context_sources(prompt: str) -> List[Dict[str, str]]:
    context_match = re.search(
        r"Ingested context:\n(?P<context>.*?)(?:\n\nOutput requirements:|\Z)",
        prompt or "",
        flags=re.DOTALL,
    )
    if not context_match:
        return []

    context = context_match.group("context").strip()
    pattern = re.compile(
        r"Source:\s*(?P<title>.*?)\n(?P<content>.*?)(?=\n\nSource:\s|\Z)",
        flags=re.DOTALL,
    )
    sources: List[Dict[str, str]] = []
    for match in pattern.finditer(context):
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        content = re.sub(r"\s+", " ", match.group("content")).strip()
        if title and content:
            sources.append({"title": title, "content": content})
    return sources


def _synthesize_ingested_points(sources: List[Dict[str, str]], user_prompt: str, max_points: int = 4) -> List[str]:
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9_]+", (user_prompt or "").lower())
        if len(token) > 3 and token not in {"from", "ingested", "folder", "what", "where", "which", "module"}
    }
    path_context_terms = {"ghostiso", "master", "perseus", "princess", "protocol", "snap", "wrap"}
    ranking_terms = query_terms.difference(path_context_terms) or query_terms

    ranked_sources = sorted(
        sources,
        key=lambda source: _ingested_source_score(source, ranking_terms),
        reverse=True,
    )
    points: List[str] = []
    seen = set()

    for source in ranked_sources[:8]:
        if ranking_terms and _ingested_source_score(source, ranking_terms) <= 0:
            continue
        title = source["title"]
        content = source["content"]
        function_points = _function_points_from_context(content)
        candidates = function_points or _sentence_points_from_context(content, ranking_terms)

        for candidate in candidates:
            point = _shorten_lookup_point(candidate, max_chars=260)
            if not point:
                continue
            key = re.sub(r"\W+", "", f"{title} {point}".lower())[:160]
            if key in seen:
                continue
            seen.add(key)
            points.append(f"`{title}`: {point}")
            break

        if len(points) >= max_points:
            break
    return points


def _ingested_source_score(source: Dict[str, str], query_terms: set) -> int:
    haystack = f"{source.get('title', '')}\n{source.get('content', '')}".lower()
    return sum(len(re.findall(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack)) for term in query_terms)


def _function_points_from_context(content: str) -> List[str]:
    points: List[str] = []
    pattern = re.compile(
        r"def\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<args>[^)]*)\):\s*(?:[rubfRUBF]*[\"']{3}(?P<doc>.*?)[\"']{3})?",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(content or ""):
        name = match.group("name")
        args = re.sub(r"\s+", " ", match.group("args")).strip()
        doc = re.sub(r"\s+", " ", match.group("doc") or "").strip()
        if doc:
            first_sentence = re.split(r"(?<=[.!?])\s+", doc)[0]
            points.append(f"defines `{name}({args})`; {first_sentence}")
        else:
            points.append(f"defines `{name}({args})` for the behavior shown in the retrieved source excerpt")
    return points


def _sentence_points_from_context(content: str, query_terms: set) -> List[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", content or "") if len(sentence.strip()) >= 35]
    ranked = sorted(
        sentences,
        key=lambda sentence: len(query_terms.intersection(set(re.findall(r"[a-z0-9_]+", sentence.lower())))),
        reverse=True,
    )
    return ranked[:2]


def _answer_from_online_search_context(prompt: str, user_prompt: str) -> str:
    """Create a compact source-grounded answer when only the local fallback provider is available."""
    results = _parse_online_search_results(prompt)
    if not results:
        return (
            "I tried an online lookup, but the search context did not return usable source snippets. "
            "Try a more specific location, source, or topic so I can look up a narrower answer."
        )

    weather_like = "weather" in user_prompt.lower() or "forecast" in user_prompt.lower()
    if weather_like:
        weather_answer = _synthesize_weather_lookup(results, user_prompt)
        if weather_answer:
            return weather_answer

    synthesized_points = _synthesize_lookup_points(results, user_prompt)
    if not synthesized_points:
        return (
            "I found online results, but their snippets were too thin to answer confidently. "
            "Try narrowing the request or asking for a specific source."
        )

    source_note = _format_source_note(results)

    return (
        f"Answer:\n{chr(10).join(f'- {point}' for point in synthesized_points)}\n\n"
        f"Sources consulted: {source_note}.\n\n"
        "Confidence note: I synthesized the online lookup evidence rather than showing the raw retrieved text. "
        "If this is high-stakes or very current, verify against the primary source."
    )


def _synthesize_weather_lookup(results: List[Dict[str, str]], user_prompt: str) -> str:
    """Turn the structured wttr.in weather snippet into a direct answer without exposing raw context."""
    for item in results:
        snippet = item.get("snippet", "")
        match = re.search(
            r"Current weather for (?P<area>.*?): (?P<condition>.*?), (?P<temp>-?\d+|\?)\D*F, "
            r"feels like (?P<feels>-?\d+|\?)\D*F, humidity (?P<humidity>\d+|\?)%, wind (?P<wind>\d+|\?) mph\. "
            r"Observation time: (?P<observed>.+?)\.?$",
            snippet,
            flags=re.I,
        )
        if not match:
            continue

        area = match.group("area").strip()
        condition = match.group("condition").strip()
        temp = match.group("temp").strip()
        feels = match.group("feels").strip()
        humidity = match.group("humidity").strip()
        wind = match.group("wind").strip()
        observed = match.group("observed").strip()
        return (
            f"Current snapshot for `{user_prompt}`:\n"
            f"- {area}: {condition}, about {temp}°F, feeling like {feels}°F.\n"
            f"- Humidity is around {humidity}% with wind near {wind} mph.\n"
            f"- Observed: {observed}. Weather changes quickly, so treat this as a live snapshot.\n\n"
            f"Sources consulted: {_format_source_note([item])}."
        )
    return ""


def _synthesize_lookup_points(results: List[Dict[str, str]], user_prompt: str, max_points: int = 3) -> List[str]:
    """Create concise evidence-based points without dumping the raw lookup snippets."""
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9']+", (user_prompt or "").lower())
        if len(token) > 3 and token not in {"what", "when", "where", "which", "about", "lookup", "search", "online"}
    }
    points: List[str] = []
    seen = set()
    for item in results[:5]:
        snippet = _clean_lookup_text(item.get("snippet", ""))
        title = _clean_lookup_text(item.get("title", ""))
        if not snippet:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", snippet)
        ranked = sorted(
            [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 30],
            key=lambda sentence: len(query_terms.intersection(set(re.findall(r"[a-z0-9']+", sentence.lower())))),
            reverse=True,
        )
        chosen = ranked[0] if ranked else snippet
        chosen = _shorten_lookup_point(chosen)
        if not chosen:
            continue

        domain = _domain_from_lookup_item(item)
        prefix = f"{title}: " if title and title.lower() not in chosen.lower() else ""
        point = f"{prefix}{chosen}"
        if domain:
            point = f"{point} ({domain})"
        key = re.sub(r"\W+", "", point.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        points.append(point)
        if len(points) >= max_points:
            break
    return points


def _clean_lookup_text(text: str) -> str:
    text = re.sub(r"\s+", " ", unescape(text or "")).strip()
    return text.strip(" \t\r\n-•")


def _shorten_lookup_point(text: str, max_chars: int = 220) -> str:
    text = _clean_lookup_text(text)
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].strip(" ,;:-")
    return f"{shortened}..." if shortened else ""


def _format_source_note(results: List[Dict[str, str]], max_sources: int = 4) -> str:
    labels: List[str] = []
    seen = set()
    for item in results:
        label = _domain_from_lookup_item(item) or item.get("source", "online source")
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= max_sources:
            break
    return ", ".join(labels) or "online lookup providers"


def _format_online_search_context(results: List[Dict[str, str]], query: str = "") -> str:
    """Format search results for internal synthesis in the same shape the fallback parser expects."""
    lines = ["ONLINE SEARCH CONTEXT"]
    if query:
        lines.append(f"Query: {query}")
    lines.append("Instruction: Use these snippets as private evidence. Do not reveal or paste this block.")
    lines.append("")
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item.get('title', 'Untitled')}")
        lines.append(f"   Source: {item.get('source', 'online')}")
        lines.append(f"   URL: {item.get('url', '')}")
        lines.append(f"   Retrieved: {item.get('retrieved', '')}")
        lines.append(f"   Snippet: {_shorten_lookup_point(item.get('snippet', ''), max_chars=700)}")
        lines.append("")
    return "\n".join(lines).strip()


def _domain_from_lookup_item(item: Dict[str, str]) -> str:
    url = item.get("url", "")
    domain = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
    if domain:
        return domain
    source = item.get("source", "")
    return source.split(" via ", 1)[0].strip()


def _parse_online_search_results(prompt: str) -> List[Dict[str, str]]:
    """Parse SearchAugmentation's compact context block into title/source/snippet items."""
    text = prompt or ""
    pattern = re.compile(
        r"^\d+\.\s*(?P<title>.*?)\n"
        r"\s*Source:\s*(?P<source>.*?)\n"
        r"\s*URL:\s*(?P<url>.*?)\n"
        r"\s*Retrieved:\s*(?P<retrieved>.*?)\n"
        r"\s*Snippet:\s*(?P<snippet>.*?)(?=\n\d+\.\s|\nInstruction:|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    results: List[Dict[str, str]] = []
    for match in pattern.finditer(text):
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        source = re.sub(r"\s+", " ", match.group("source")).strip()
        snippet = re.sub(r"\s+", " ", match.group("snippet")).strip()
        if title and snippet:
            results.append({"title": title, "source": source, "snippet": snippet, "url": match.group("url").strip()})
    return results


def _is_memory_prompt(prompt: str) -> bool:
    """Return True when the user is asking Perseus to use or update learned chat memory."""
    text = _extract_user_request(prompt)
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .!?\t\r\n")
    memory_markers = [
        "remember",
        "what do you know about me",
        "what do you remember about me",
        "what have you learned",
        "what did you learn",
        "my preferences",
        "my project context",
        "user memory",
        "chat memory",
        "learn as you go",
        "learned about me",
    ]
    return any(marker in lower for marker in memory_markers)


def _is_general_knowledge_prompt(prompt: str) -> bool:
    """Return True for short general-purpose questions that should not require ingested context."""
    text = _extract_user_request(prompt)
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .!?\t\r\n")
    if not lower:
        return False

    if _is_memory_prompt(prompt):
        return False

    if _general_knowledge_fallback(lower):
        return True

    question_starters = (
        "is ",
        "are ",
        "do ",
        "does ",
        "can ",
        "could ",
        "should ",
        "would ",
        "who ",
        "when ",
        "where ",
        "which ",
        "what ",
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
        "remember",
        "memory",
        "preferences",
        "knowledge base",
        "file",
        "folder",
        "codebase",
    ]
    return lower.startswith(question_starters) and len(lower.split()) <= 24 and not any(
        marker in lower for marker in domain_markers
    )


def _context_preview_bullets(context: str, max_bullets: int = 6) -> List[str]:
    """Turn retrieved context preview text into concise fallback bullets."""
    text = re.sub(r"\s+", " ", (context or "").strip())
    if not text:
        return []

    parts = re.split(r"(?:(?:Source:|Memory summary:|Retrieval keywords:)\s*)", text)
    candidates: List[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" -:;.,")
        if not cleaned:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
            sentence = sentence.strip(" -:;.,")
            if len(sentence) < 18:
                continue
            candidates.append(sentence[:220])
            break

    if not candidates:
        candidates = [text[:220]]

    bullets: List[str] = []
    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(candidate)
        if len(bullets) >= max_bullets:
            break
    return bullets


def _format_prediction_packet(label: str, packet: Dict[str, object]) -> str:
    """Format predictive/asynchronous module output for the LLM prompt."""
    if not packet:
        return ""
    prediction = str(packet.get("prediction") or "").strip()
    confidence = packet.get("confidence", "")
    lessons = [str(item).strip() for item in packet.get("relevant_lessons", []) if str(item).strip()]
    suggested = str(packet.get("suggested_reasoning") or "").strip()
    related = packet.get("related_events") or []

    lines = [
        f"{label} CONTEXT",
        "Use as probabilistic background memory, not absolute truth.",
        f"Prediction: {prediction}",
        f"Confidence: {confidence}",
    ]
    if lessons:
        lines.append("Relevant lessons:")
        lines.extend(f"- {lesson[:300]}" for lesson in lessons[:5])
    if related:
        lines.append("Related prior events:")
        for event in related[:3]:
            what = str(event.get("what") or "").strip()
            outcome = str(event.get("outcome") or "").strip()
            if what:
                lines.append(f"- Prior what: {what[:220]}")
            if outcome:
                lines.append(f"  Outcome: {outcome[:220]}")
    if suggested:
        lines.append(f"Suggested reasoning: {suggested[:300]}")
    return "\n".join(lines)


def _format_cognitive_traces(traces: List[object]) -> str:
    """Format Cognitive Functions memory traces for the prompt."""
    if not traces:
        return ""
    lines = [
        "COGNITIVE FUNCTIONS CONTEXT",
        "Use prior cognitive traces for self-review, assumptions, risks, and response coherence.",
    ]
    for trace in traces[:5]:
        lesson = str(getattr(trace, "lesson", "") or "").strip()
        why = str(getattr(trace, "why_response_was_given", "") or "").strip()
        tone = str(getattr(trace, "emotional_tone", "") or "").strip()
        confidence = getattr(trace, "confidence", "")
        risks = [str(item).strip() for item in (getattr(trace, "risks", []) or []) if str(item).strip()]
        if lesson:
            lines.append(f"- Lesson: {lesson[:260]}")
        if why:
            lines.append(f"  Why prior response happened: {why[:260]}")
        if tone or confidence != "":
            lines.append(f"  Tone/confidence: {tone or 'unknown'} / {confidence}")
        for risk in risks[:2]:
            lines.append(f"  Risk note: {risk[:220]}")
    return "\n".join(lines)


def _predictive_lesson_for_turn(profile: PromptProfile, quality: ResponseQuality, memories: List[str]) -> str:
    """Create a reusable predictive lesson from the response outcome."""
    parts: List[str] = []
    if quality.score >= 85:
        parts.append(
            f"For {profile.intent} prompts, this response pattern worked well: preserve depth, structure, evidence, and concrete next steps."
        )
    elif quality.score >= 72:
        parts.append(
            f"For {profile.intent} prompts, the response was acceptable but should keep improving specificity, visible rationale, and privacy-safe explanations."
        )
    else:
        parts.append(
            f"For {profile.intent} prompts, avoid repeating this weak pattern; address quality issues before finalizing."
        )

    if quality.reasons:
        parts.append("Quality signals: " + "; ".join(quality.reasons[:5]))
    if memories:
        parts.append("User/context lessons: " + " | ".join(memories[:4]))
    return " ".join(parts)[:900]


def _looks_sensitive_memory(text: str) -> bool:
    """Avoid persisting obvious secrets as durable memory."""
    lower = (text or "").lower()
    if any(marker in lower for marker in SENSITIVE_MEMORY_MARKERS):
        return True
    secret_like_patterns = [
        r"sk-[a-zA-Z0-9]{16,}",
        r"[a-zA-Z0-9_=-]{32,}",
        r"-----BEGIN [A-Z ]+PRIVATE KEY-----",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b(?:\d[ -]*?){13,16}\b",
    ]
    return any(re.search(pattern, text or "") for pattern in secret_like_patterns)


def _looks_low_value_memory(text: str) -> bool:
    """Filter memories that are too vague to help future retrieval or behavior."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if len(normalized) < 12:
        return True
    low_value = {
        "user preference: help",
        "user preference: this",
        "user preference: that",
        "user self-context: here",
        "user self-context: ready",
        "user explicitly asked perseus to remember: this",
    }
    if normalized in low_value:
        return True
    content_terms = [term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", normalized) if term not in MEMORY_RETRIEVAL_STOPWORDS]
    return len(content_terms) < 2


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
        return _compose_question_answer(
            focus=subject,
            direct_answer=(
                f"{subject.capitalize()} exist because they either solve a problem, emerge from a cause, or persist "
                "because people, systems, or environments keep selecting for them."
            ),
            why="They continue when their usefulness, incentives, or natural causes outweigh the cost of removing them.",
            how="To explain them well, identify the need they satisfy, the conditions that made them possible, and the feedback loop that keeps them around.",
            what="The key object of analysis is not just the thing itself, but the function or pressure behind it.",
        )

    why_is_match = re.fullmatch(r"why is (.+) ([a-z][a-z-]*)", lower)
    if why_is_match:
        subject = why_is_match.group(1).strip()
        trait = why_is_match.group(2).strip()
        return _compose_question_answer(
            focus=subject,
            direct_answer=(
                f"{subject.capitalize()} is {trait} because some underlying property, process, or context produces "
                f"that {trait} effect."
            ),
            why="The cause usually sits one layer deeper than the description: material, structure, incentive, biology, physics, or social convention.",
            how="Break it into inputs, mechanism, and visible result: what acts on it, how it changes, and what you observe.",
            what=f"The trait to explain is `{trait}`; the useful answer should separate appearance from mechanism.",
        )

    why_do_match = re.fullmatch(r"why (?:do|does) (.+?) (.+)", lower)
    if why_do_match:
        subject = why_do_match.group(1).strip()
        action = why_do_match.group(2).strip()
        return _compose_question_answer(
            focus=subject,
            direct_answer=(
                f"{subject.capitalize()} {action} because a mechanism enables it and some pressure, payoff, or cause "
                "makes it happen repeatedly."
            ),
            why="The reason is usually a mix of cause and function: what triggers the behavior, and what result it produces.",
            how="Look for the mechanism: physical forces, biological structures, incentives, rules, tools, or habits that make the action possible.",
            what=f"The behavior to explain is `{action}`; a strong answer should connect cause -> mechanism -> outcome.",
        )

    question_match = re.fullmatch(r"(who|what|when|where|why|how|which)\s+(.+)", lower)
    if question_match:
        question_word = question_match.group(1)
        raw_focus = question_match.group(2).strip()
        focus = _normalize_question_focus(question_word, raw_focus)
        return _compose_question_answer(
            focus=focus,
            direct_answer=_direct_heuristic_answer(question_word, focus, raw_focus),
            who="Identify the actor, owner, decision-maker, affected group, or source of authority involved.",
            what="Define the object precisely before judging it; vague nouns usually create vague answers.",
            when="Time matters if the answer depends on sequence, deadlines, history, version, or changing conditions.",
            where="Place matters if jurisdiction, environment, deployment target, market, or physical context changes the answer.",
            why="The deeper value is the cause, incentive, risk, or purpose behind the surface fact.",
            how="A sophisticated answer connects the mechanism to practical consequences: what causes what, and what to do with that information.",
        )

    return _compose_question_answer(
        focus=lower or "the question",
        direct_answer=(
            "The intelligent way to answer this is to define the subject, identify the mechanism, separate facts from assumptions, "
            "and then explain why the answer matters in practice."
        ),
        what="Clarify the exact subject and the decision or curiosity behind it.",
        why="The purpose of the question determines whether the answer should be factual, strategic, diagnostic, or explanatory.",
        how="Once the goal is clear, answer by connecting evidence -> mechanism -> implication -> next action.",
    )


def _normalize_question_focus(question_word: str, focus: str) -> str:
    """Remove leading auxiliaries so fallback answers discuss the actual topic."""
    cleaned = focus.rstrip("?").strip()
    cleaned = re.sub(r"^(?:do|does|did|is|are|was|were|can|could|should|would)\s+", "", cleaned)
    cleaned = re.sub(r"^(?:i|we|you|they)\s+", "", cleaned)
    if question_word == "how" and cleaned.endswith(" work"):
        cleaned = cleaned[: -len(" work")].strip()
    if question_word == "where" and cleaned.startswith("do "):
        cleaned = cleaned[3:].strip()
    return cleaned or focus.strip() or "the question"


def _direct_heuristic_answer(question_word: str, focus: str, raw_focus: str = "") -> str:
    """Create a cautious direct answer when the fallback has no factual knowledge base for the topic."""
    cleaned = focus.rstrip("?").strip()
    raw = raw_focus.lower()
    if question_word == "who" and "invented the telephone" in cleaned:
        return (
            "Alexander Graham Bell is commonly credited with inventing and patenting the practical telephone in 1876, "
            "though the broader story includes competing inventors and contributors such as Elisha Gray and Antonio Meucci."
        )
    if question_word == "who":
        return (
            f"For `{cleaned}`, the useful first move is to identify the relevant person, group, institution, or role, "
            "then verify it against a primary source before treating it as fact."
        )
    if question_word == "when":
        if raw.startswith("should") or raw.startswith("could") or raw.startswith("would"):
            return (
                f"For `{cleaned}`, the answer is conditional: do it when the benefit is clear, the constraints are understood, "
                "and the downside is manageable."
            )
        return (
            f"For `{cleaned}`, the answer depends on the timeline: origin, trigger point, deadline, and whether the date changes by location or version."
        )
    if question_word == "where":
        return (
            f"For `{cleaned}`, the important location may be physical, legal, technical, or organizational; the context can change the answer."
        )
    if question_word == "how":
        return (
            f"For `{cleaned}`, focus on the mechanism: inputs, process, constraints, output, and failure modes."
        )
    if question_word == "why":
        return (
            f"For `{cleaned}`, the best answer should identify the underlying cause, incentive, or purpose rather than only describing the surface fact."
        )
    return (
        f"For `{cleaned}`, start by defining the subject precisely, then separate known facts, assumptions, implications, and useful next actions."
    )


def _compose_question_answer(
    focus: str,
    direct_answer: str,
    who: str = "",
    what: str = "",
    when: str = "",
    where: str = "",
    why: str = "",
    how: str = "",
) -> str:
    """Format fallback answers with explicit question decomposition and synthesis."""
    dimensions = [
        ("Who", who),
        ("What", what),
        ("When", when),
        ("Where", where),
        ("Why", why),
        ("How", how),
    ]
    relevant = [(label, value) for label, value in dimensions if value]
    dimension_lines = "\n".join(f"- {label}: {value}" for label, value in relevant)
    if not dimension_lines:
        dimension_lines = "- Why: Identify the cause or purpose.\n- How: Explain the mechanism and practical effect."

    return (
        f"Direct answer: {direct_answer}\n\n"
        f"Question anatomy for `{focus}`:\n"
        f"{dimension_lines}\n\n"
        "Synthesis: a sophisticated answer should not stop at the surface fact. It should connect cause, mechanism, context, "
        "and implication so you know not just the answer, but how to reason about it and what to do next."
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
