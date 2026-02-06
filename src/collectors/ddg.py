"""DuckDuckGo web and news search collectors (free, no API key)."""

import asyncio
from typing import Any

from duckduckgo_search import DDGS

from src.collectors.base import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class DdgWebCollector(BaseCollector):
    """DuckDuckGo web search — free, no API key required."""

    def __init__(self):
        super().__init__(name="ddg_web")

    def _search_sync(self, query: str, limit: int) -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=limit))

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        limit = kwargs.get("limit", 10)
        results = await asyncio.to_thread(self._search_sync, query, limit)

        items: list[CollectedItem] = []
        for r in results:
            items.append(
                CollectedItem(
                    source="ddg_web",
                    title=r.get("title", ""),
                    content=r.get("body", ""),
                    url=r.get("href", ""),
                )
            )
        return items

    async def close(self) -> None:
        pass  # DDGS manages its own HTTP


class DdgNewsCollector(BaseCollector):
    """DuckDuckGo news search — free, no API key required."""

    def __init__(self):
        super().__init__(name="ddg_news")

    def _search_news_sync(self, query: str, limit: int) -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.news(query, max_results=limit))

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        limit = kwargs.get("limit", 10)
        results = await asyncio.to_thread(self._search_news_sync, query, limit)

        items: list[CollectedItem] = []
        for r in results:
            items.append(
                CollectedItem(
                    source="ddg_news",
                    title=r.get("title", ""),
                    content=r.get("body", ""),
                    url=r.get("url", ""),
                    published_at=r.get("date", ""),
                    metadata={"news_source": r.get("source", "")},
                )
            )
        return items

    async def close(self) -> None:
        pass
