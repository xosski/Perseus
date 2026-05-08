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

from dataclasses import dataclass
from html import unescape
import logging
import queue
import re
import sqlite3
import threading
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

try:
    from llm_conversation_core import ConversationManager
    from offline_llm import OfflineLLM
except ImportError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from llm_conversation_core import ConversationManager
    from offline_llm import OfflineLLM

try:
    from web_knowledge_learner import WebKnowledgeLearner
except ImportError:
    WebKnowledgeLearner = None

WEB_LEARNING_DB_PATH = "llm_web_learning.db"
SELF_IMPROVEMENT_DB_PATH = "llm_self_improvement.db"

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

MAX_FEED_ITEMS_PER_SOURCE = 6


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
            "Provide accurate, structured, and actionable responses."
        ),
    ):
        self.strict_local_only = bool(strict_local_only)
        self._local_provider_order = ["ollama"]
        self.manager = ConversationManager(db_path=db_path)
        self.offline = (
            OfflineLLM(use_knowledge_db=True)
            if use_offline_fallback and not self.strict_local_only
            else None
        )
        self.system_prompt = system_prompt
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

    def ingest_news_sources(self, sources: Optional[List[str]] = None, timeout: int = 15) -> Dict[str, object]:
        """Fetch and ingest a list of news/feed sources and linked stories."""
        source_list = sources or DEFAULT_NEWS_SOURCES
        results: List[Dict[str, object]] = []
        successes = 0
        total_entries_ingested = 0
        total_entry_failures = 0

        for source in source_list:
            result = self.ingest_feed_source(source, timeout=timeout)
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

    def ingest_feed_source(self, source_url: str, timeout: int = 15) -> Dict[str, object]:
        """Ingest a feed page and a bounded set of article URLs listed in the feed."""
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
        links = links[:MAX_FEED_ITEMS_PER_SOURCE]

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
            "Ground your answer in that ingested context and be explicit when you do.\n\n"
            "Ingested context:\n"
            f"{context}\n\n"
            "Output requirements:\n"
            "1. Start with a realistic summary of what is currently known.\n"
            "2. Include an 'Ingested Context Used' section with concrete points.\n"
            "3. Include uncertainty where evidence is incomplete.\n"
            "4. Provide educational explanation and practical implications.\n"
            "5. End with a 'Next Steps' section focused on action.\n\n"
            "User request:\n"
            f"{prompt}"
        )

        return EnrichedPrompt(text=enriched_text, has_context=True, context_preview=preview)

    def _lookup_knowledge_context(self, prompt: str) -> str:
        """Try multiple targeted queries so stored knowledge is actually retrieved."""
        queries = self._knowledge_queries(prompt)
        collected: List[str] = []
        seen = set()

        for q in queries:
            if not q:
                continue
            context = self.web_learner.get_knowledge_context_for_query(q)
            snippet = (context or "").strip()
            if snippet and snippet not in seen:
                collected.append(snippet[:1200])
                seen.add(snippet)
            if len(collected) >= 3:
                break

        return "\n\n".join(collected).strip()

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

        if any(token in lower for token in ["compare", "tradeoff", "trade-off", "evaluate", "analyze", "analysis"]):
            intent = "analytical"
        elif any(token in lower for token in ["plan", "strategy", "roadmap", "prioritize", "recommend"]):
            intent = "strategic"
        elif any(token in lower for token in ["teach", "explain", "what is", "how does", "why"]):
            intent = "educational"
        else:
            intent = "technical"

        complexity = "high" if len(words) > 20 or any(t in lower for t in ["architecture", "constraints", "production", "distributed"]) else "normal"
        mood = "analytical" if intent in {"technical", "analytical"} else "pragmatic"
        conversational_markers = ["chat", "talk", "casual", "normal", "plain english", "simple terms"]
        concise_markers = ["brief", "short", "concise", "tldr", "quick answer", "one-liner", "one line"]
        structured_markers = ["steps", "plan", "outline", "table", "bullet", "checklist", "roadmap", "compare"]

        prefer_concise = any(marker in lower for marker in concise_markers)
        prefer_structure = any(marker in lower for marker in structured_markers) or intent in {"strategic", "analytical"}
        conversational = any(marker in lower for marker in conversational_markers) or not prefer_structure

        if prefer_structure:
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
            "Respond like a capable general-purpose assistant: natural, clear, and directly useful.",
            "Match the user's tone and requested depth.",
            "If uncertain, state assumptions and what data is needed.",
        ]

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
                "\nImprove the previous draft by increasing specificity, correctness, and practical detail. "
                "Remove generic phrasing and tighten structure."
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
) -> None:
    """Launch a desktop chat window for PortableLLM (no terminal loop)."""
    import tkinter as tk
    from tkinter import ttk
    import webbrowser

    llm = PortableLLM(
        db_path=db_path,
        provider=provider,
        model=model,
        strict_local_only=strict_local_only,
        use_offline_fallback=use_offline_fallback,
    )
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
    tabs.add(ingest_tab, text="Web Ingest")

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
        text="Auto ingest news sources on start",
        variable=auto_ingest_var,
    )
    auto_ingest_check.pack(side=tk.LEFT)

    ingest_status_var = tk.StringVar(value="Ingest ready")
    ingest_status = ttk.Label(ingest_tab, textvariable=ingest_status_var)
    ingest_status.pack(fill=tk.X)

    sources_label = ttk.Label(ingest_tab, text="News Sources (one URL per line):")
    sources_label.pack(anchor=tk.W)

    sources_box = tk.Text(ingest_tab, height=6, wrap=tk.WORD, font=("Consolas", 10))
    sources_box.pack(fill=tk.X, pady=(4, 8))
    sources_box.insert("1.0", "\n".join(DEFAULT_NEWS_SOURCES))

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
        result = llm.ingest_news_sources(sources=sources, timeout=timeout)
        ingest_results_queue.put(("bulk", result))

    def worker_ingest_manual(url: str, timeout: int) -> None:
        result = llm.ingest_url(url, timeout=timeout)
        ingest_results_queue.put(("manual", result))

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
                f"Auto ingest complete: {payload.get('successes')}/{payload.get('total')} feeds succeeded"
            )
            append_ingest(
                (
                    f"Auto ingest complete: {payload.get('successes')}/{payload.get('total')} feeds succeeded"
                    f" | entries={payload.get('entry_successes', 0)} ok/{payload.get('entry_failures', 0)} fail"
                )
            )
            for item in payload.get("results", []):
                if item.get("ok"):
                    learned = item.get("summary", {}).get("total_items_learned", 0)
                    append_ingest(
                        (
                            f"[OK] {item.get('url')} | title={item.get('title', '')}"
                            f" | feed_learned={learned}"
                            f" | entries={item.get('entry_successes', 0)}/{item.get('entries_discovered', 0)}"
                        )
                    )
                else:
                    append_ingest(f"[FAIL] {item.get('url')} | error={item.get('error')}")
        else:
            if payload.get("ok"):
                learned = payload.get("summary", {}).get("total_items_learned", 0)
                ingest_status_var.set("Manual ingest succeeded")
                append_ingest(f"[OK] {payload.get('url')} | title={payload.get('title', '')} | learned={learned}")
            else:
                ingest_status_var.set("Manual ingest failed")
                append_ingest(f"[FAIL] {payload.get('url')} | error={payload.get('error')}")

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
        timeout = _timeout_value()
        ingest_status_var.set("Auto ingest in progress...")
        append_ingest(f"Auto ingest started with {len(sources)} sources (timeout={timeout}s)")
        threading.Thread(target=worker_ingest_sources, args=(sources, timeout), daemon=True).start()

    send_btn = ttk.Button(controls, text="Send", command=send_message)
    send_btn.pack(side=tk.LEFT, padx=(8, 0))

    clear_btn = ttk.Button(controls, text="Clear", command=lambda: transcript.delete("1.0", tk.END))
    clear_btn.pack(side=tk.LEFT, padx=(8, 0))

    ingest_controls = ttk.Frame(ingest_tab)
    ingest_controls.pack(fill=tk.X, pady=(8, 8))

    ingest_now_btn = ttk.Button(ingest_controls, text="Ingest News Sources Now", command=ingest_now)
    ingest_now_btn.pack(side=tk.LEFT)

    ingest_manual_btn = ttk.Button(ingest_controls, text="Ingest Manual URL", command=ingest_manual)
    ingest_manual_btn.pack(side=tk.LEFT, padx=(8, 0))

    input_box.bind("<Return>", send_message)
    manual_url_entry.bind("<Return>", lambda _event: ingest_manual())

    append_block("SYSTEM", "Perseus chat window is ready.")
    append_ingest("Web ingest tab ready.")
    root.after(80, poll_results)
    root.after(100, poll_ingest_results)
    input_box.focus_set()

    if auto_ingest_var.get():
        ingest_now()

    root.after(250, show_donation_popup)

    def on_close() -> None:
        llm.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


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
