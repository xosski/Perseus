#!/usr/bin/env python3
"""
Search Augmentation Module for PortableLLM / Perseus.

Purpose:
- Decide when online search is useful.
- Query search providers or fallback sources.
- Cache results locally.
- Return compact source context to PortableLLM.

Network providers:
- Brave Search API if BRAVE_SEARCH_API_KEY is set.
- Serper API if SERPER_API_KEY is set.
- Wikipedia summary fallback for broad stable topics.
- DuckDuckGo HTML fallback as last resort.

Set PortableLLM(strict_local_only=False) to allow this module to use network access.
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

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_cached(self, query: str) -> Optional[List[SearchResult]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT created_epoch, results_json FROM search_cache WHERE query = ?",
                (query,),
            ).fetchone()

        if not row:
            return None

        if int(time.time()) - int(row["created_epoch"] or 0) > self.cache_ttl_seconds:
            return None

        try:
            return [SearchResult(**item) for item in json.loads(row["results_json"] or "[]")]
        except Exception:
            return None

    def _set_cached(self, query: str, results: List[SearchResult]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO search_cache
                (query, created_utc, created_epoch, results_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    query,
                    self._now_utc(),
                    int(time.time()),
                    json.dumps([asdict(result) for result in results], ensure_ascii=False),
                ),
            )
            conn.commit()

    def should_search(
        self,
        prompt: str,
        local_context: str = "",
        draft_response: str = "",
        quality_score: Optional[int] = None,
    ) -> SearchDecision:
        prompt_clean = (prompt or "").strip()
        prompt_lower = prompt_clean.lower()
        draft_lower = (draft_response or "").lower()

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

        explicit_terms = [
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
        if any(term in prompt_lower for term in explicit_terms):
            return SearchDecision(True, "User explicitly requested current or online information.", 0.95)

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
        if any(term in prompt_lower for term in current_sensitive_terms):
            return SearchDecision(True, "Question may depend on current information.", 0.8)

        if quality_score is not None and quality_score < 60:
            return SearchDecision(True, "Draft quality score is low.", 0.75)

        weak_markers = [
            "i do not have enough",
            "not enough learned context",
            "i don't know",
            "cannot answer with confidence",
            "add trusted sites",
            "ingest source material",
        ]
        if any(marker in draft_lower for marker in weak_markers):
            return SearchDecision(True, "Draft response says local knowledge is insufficient.", 0.85)

        return SearchDecision(False, "Local/model answer should be sufficient.", 0.7)

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
        items = ((data or {}).get("web") or {}).get("results") or []
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
        items = (data or {}).get("organic") or []
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
        topic = re.sub(r"^(tell me about|what is|who is|explain)\s+", "", query.strip(), flags=re.I)
        topic = topic.strip(" ?.!")

        if not topic:
            return []

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(topic)}"
        data = self._request_json(url, headers={"Accept": "application/json"})
        if not data or "not_found" in str(data.get("type", "")):
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
        html = self._request_text(f"https://duckduckgo.com/html/?q={quote_plus(query)}")
        if not html:
            return []

        now = self._now_utc()
        results: List[SearchResult] = []
        blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>.*?'
            r'<a class="result__snippet".*?>(.*?)</a>',
            html,
            flags=re.DOTALL | re.I,
        )

        for url, title, snippet in blocks[: self.max_results]:
            results.append(
                SearchResult(
                    title=self._strip_html(title),
                    url=unescape(url),
                    snippet=self._strip_html(snippet),
                    source="duckduckgo_html",
                    retrieved_utc=now,
                )
            )
        return results

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
        return self.build_search_context(query, self.search(query))

    @staticmethod
    def _strip_html(text: str) -> str:
        text = re.sub(r"<.*?>", " ", text or "", flags=re.DOTALL)
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _dedupe_results(self, results: List[SearchResult]) -> List[SearchResult]:
        seen = set()
        clean: List[SearchResult] = []
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

    @staticmethod
    def _canonical_url(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.netloc.lower()}{parsed.path}".rstrip("/")
