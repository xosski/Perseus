#!/usr/bin/env python3
"""
Search Augmentation Module for PortableLLM / Perseus

Purpose:
- Detect when local knowledge is insufficient.
- Search the web using configurable providers.
- Cache search results locally.
- Return compact source-grounded context to the LLM.
- Avoid hijacking simple prompts when local answer is enough.

Supported providers:
- Brave Search API, if BRAVE_SEARCH_API_KEY is set.
- Serper Google Search API, if SERPER_API_KEY is set.
- Wikipedia summary fallback.
- DuckDuckGo HTML fallback.

This module does not generate final answers.
It returns evidence/context for the main LLM to use.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from html import unescape
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


DEFAULT_DB_PATH = "llm_search_cache.db"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    retrieved_utc: str


@dataclass
class SearchDecision:
    should_search: bool
    reason: str
    confidence: float


class SearchAugmentation:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        max_results: int = 5,
        timeout_seconds: int = 8,
        cache_ttl_seconds: int = 60 * 60 * 24,
        allow_network: bool = True,
    ):
        self.db_path = db_path
        self.max_results = int(max_results)
        self.timeout_seconds = int(timeout_seconds)
        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self.allow_network = bool(allow_network)

        self.brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        self.serper_key = os.getenv("SERPER_API_KEY", "").strip()

        self._init_db()

    # -------------------------
    # DB / cache
    # -------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                query TEXT PRIMARY KEY,
                created_utc TEXT,
                created_epoch INTEGER,
                results_json TEXT
            )
            """)
            conn.commit()

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_cached(self, query: str) -> Optional[List[SearchResult]]:
        now = int(time.time())

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT created_epoch, results_json FROM search_cache WHERE query = ?",
                (query,),
            ).fetchone()

        if not row:
            return None

        age = now - int(row["created_epoch"] or 0)
        if age > self.cache_ttl_seconds:
            return None

        try:
            payload = json.loads(row["results_json"] or "[]")
            return [SearchResult(**item) for item in payload]
        except Exception:
            return None

    def _set_cached(self, query: str, results: List[SearchResult]) -> None:
        payload = json.dumps([asdict(r) for r in results], ensure_ascii=False)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO search_cache
                (query, created_utc, created_epoch, results_json)
                VALUES (?, ?, ?, ?)
                """,
                (query, self._now_utc(), int(time.time()), payload),
            )
            conn.commit()

    # -------------------------
    # Search decision
    # -------------------------

    def should_search(
        self,
        prompt: str,
        local_context: str = "",
        draft_response: str = "",
        quality_score: Optional[int] = None,
    ) -> SearchDecision:
        """
        Decide if online search is needed.

        Search when:
        - user explicitly asks to search/look up/current/latest
        - the question depends on current facts
        - local context is absent or weak
        - draft response admits insufficient knowledge
        - quality score is low

        Do not search for:
        - small talk
        - pure creative writing
        - local file questions
        - simple stable facts unless local/model answer is missing
        """

        prompt_clean = (prompt or "").strip()
        prompt_lower = prompt_clean.lower()
        context_lower = (local_context or "").lower()
        draft_lower = (draft_response or "").lower()

        explicit_search_terms = [
            "search the web",
            "look up",
            "google",
            "browse",
            "latest",
            "current",
            "recent",
            "today",
            "right now",
            "news",
            "price",
            "schedule",
            "weather",
            "stock",
            "update",
            "verify online",
        ]

        current_sensitive_terms = [
            "ceo",
            "president",
            "prime minister",
            "law",
            "regulation",
            "version",
            "release",
            "patch",
            "vulnerability",
            "cve",
            "market",
            "price",
            "election",
            "war",
            "sports",
            "forecast",
        ]

        no_search_patterns = [
            r"^hi\b",
            r"^hello\b",
            r"^hey\b",
            r"write me a poem",
            r"rewrite this",
            r"translate this",
            r"summarize this text",
        ]

        if any(re.search(pattern, prompt_lower) for pattern in no_search_patterns):
            return SearchDecision(False, "Prompt does not require online lookup.", 0.9)

        if any(term in prompt_lower for term in explicit_search_terms):
            return SearchDecision(True, "User explicitly asked for current or online information.", 0.95)

        if any(term in prompt_lower for term in current_sensitive_terms):
            return SearchDecision(True, "Question may depend on current information.", 0.8)

        if quality_score is not None and quality_score < 60:
            return SearchDecision(True, "Draft quality score is low.", 0.75)

        weak_response_markers = [
            "i do not have enough",
            "not enough learned context",
            "i don't know",
            "cannot answer with confidence",
            "add trusted sites",
            "ingest source material",
        ]

        if any(marker in draft_lower for marker in weak_response_markers):
            return SearchDecision(True, "Draft response says local knowledge is insufficient.", 0.85)

        if not context_lower and len(prompt_clean.split()) >= 5:
            # Mild signal only. Do not search every normal question.
            if prompt_lower.startswith(("what is", "who is", "tell me about", "explain", "how does")):
                return SearchDecision(False, "Likely stable general knowledge; local/model answer should be tried first.", 0.65)

        return SearchDecision(False, "Local/model answer should be sufficient.", 0.7)

    # -------------------------
    # Search providers
    # -------------------------

    def search(self, query: str, force_refresh: bool = False) -> List[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        cached = None if force_refresh else self._get_cached(query)
        if cached is not None:
            return cached[: self.max_results]

        if not self.allow_network:
            return []

        results: List[SearchResult] = []

        if self.brave_key:
            results = self._search_brave(query)

        if not results and self.serper_key:
            results = self._search_serper(query)

        if not results:
            results = self._search_wikipedia(query)

        if not results:
            results = self._search_duckduckgo_html(query)

        cleaned = self._dedupe_results(results)[: self.max_results]
        self._set_cached(query, cleaned)
        return cleaned

    def _request_json(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        method: str = "GET",
        body: Optional[bytes] = None,
    ) -> Optional[Dict]:
        headers = headers or {}
        headers.setdefault("User-Agent", "PerseusPortableLLM/1.0")

        try:
            req = Request(url, data=body, headers=headers, method=method)
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except Exception:
            return None

    def _request_text(self, url: str, headers: Optional[Dict[str, str]] = None) -> str:
        headers = headers or {}
        headers.setdefault("User-Agent", "PerseusPortableLLM/1.0")

        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _search_brave(self, query: str) -> List[SearchResult]:
        url = f"https://api.search.brave.com/res/v1/web/search?q={quote_plus(query)}&count={self.max_results}"
        data = self._request_json(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_key,
            },
        )

        if not data:
            return []

        web = data.get("web") or {}
        items = web.get("results") or []
        now = self._now_utc()

        return [
            SearchResult(
                title=unescape(item.get("title") or ""),
                url=item.get("url") or "",
                snippet=unescape(item.get("description") or ""),
                source="brave",
                retrieved_utc=now,
            )
            for item in items
            if item.get("url")
        ]

    def _search_serper(self, query: str) -> List[SearchResult]:
        body = json.dumps({"q": query, "num": self.max_results}).encode("utf-8")
        data = self._request_json(
            "https://google.serper.dev/search",
            method="POST",
            body=body,
            headers={
                "X-API-KEY": self.serper_key,
                "Content-Type": "application/json",
            },
        )

        if not data:
            return []

        items = data.get("organic") or []
        now = self._now_utc()

        return [
            SearchResult(
                title=unescape(item.get("title") or ""),
                url=item.get("link") or "",
                snippet=unescape(item.get("snippet") or ""),
                source="serper",
                retrieved_utc=now,
            )
            for item in items
            if item.get("link")
        ]

    def _search_wikipedia(self, query: str) -> List[SearchResult]:
        """
        Good fallback for broad factual topics.
        Not a replacement for search on current events.
        """

        # Use only the core topic. This prevents huge question strings from failing.
        topic = re.sub(r"^(tell me about|what is|who is|explain)\s+", "", query.strip(), flags=re.I)
        topic = topic.strip(" ?.!")

        if not topic:
            return []

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(topic)}"
        data = self._request_json(url, headers={"Accept": "application/json"})

        if not data or data.get("type") == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found":
            return []

        title = data.get("title") or topic
        extract = data.get("extract") or ""
        page_url = ((data.get("content_urls") or {}).get("desktop") or {}).get("page") or ""

        if not extract or not page_url:
            return []

        return [
            SearchResult(
                title=title,
                url=page_url,
                snippet=extract,
                source="wikipedia",
                retrieved_utc=self._now_utc(),
            )
        ]

    def _search_duckduckgo_html(self, query: str) -> List[SearchResult]:
        """
        Last-resort fallback. HTML scraping can break.
        Prefer Brave or Serper keys for reliability.
        """

        html = self._request_text(f"https://duckduckgo.com/html/?q={quote_plus(query)}")
        if not html:
            return []

        results: List[SearchResult] = []
        now = self._now_utc()

        # Lightweight extraction for DuckDuckGo HTML result blocks.
        blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>.*?'
            r'<a class="result__snippet".*?>(.*?)</a>',
            html,
            flags=re.DOTALL | re.I,
        )

        for url, title, snippet in blocks[: self.max_results]:
            title = self._strip_html(title)
            snippet = self._strip_html(snippet)
            url = unescape(url)

            if url:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo_html",
                        retrieved_utc=now,
                    )
                )

        return results

    # -------------------------
    # Context formatting
    # -------------------------

    def build_search_context(self, query: str, results: Optional[List[SearchResult]] = None) -> str:
        results = results if results is not None else self.search(query)

        if not results:
            return ""

        lines = [
            "ONLINE SEARCH CONTEXT",
            "Use this context carefully. Treat snippets as leads, not final truth.",
            f"Search query: {query}",
            "",
            "Results:",
        ]

        for index, result in enumerate(results, start=1):
            domain = urlparse(result.url).netloc
            lines.append(f"{index}. {result.title}")
            lines.append(f"   Source: {domain} via {result.source}")
            lines.append(f"   URL: {result.url}")
            lines.append(f"   Retrieved: {result.retrieved_utc}")
            lines.append(f"   Snippet: {result.snippet[:800]}")
            lines.append("")

        lines.append(
            "Instruction: Answer using the search context only when it is relevant. "
            "Separate confirmed facts from assumptions. Mention uncertainty when snippets are thin."
        )

        return "\n".join(lines)

    def search_and_build_context(self, query: str) -> str:
        results = self.search(query)
        return self.build_search_context(query, results)

    # -------------------------
    # Utilities
    # -------------------------

    def _strip_html(self, text: str) -> str:
        text = re.sub(r"<.*?>", " ", text or "", flags=re.DOTALL)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _dedupe_results(self, results: List[SearchResult]) -> List[SearchResult]:
        seen = set()
        clean = []

        for result in results:
            url = (result.url or "").strip()
            title = (result.title or "").strip()

            if not url or not title:
                continue

            key = self._canonical_url(url)
            if key in seen:
                continue

            seen.add(key)
            clean.append(result)

        return clean

    def _canonical_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.netloc.lower()}{parsed.path}".rstrip("/")