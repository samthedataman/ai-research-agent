"""TMZ celebrity/entertainment news via RSS feed (free, no API key)."""

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from src.collectors.base import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class TmzCollector(BaseCollector):
    """TMZ celebrity news via RSS — free, no API key required."""

    RSS_URL = "https://www.tmz.com/rss.xml"

    def __init__(self):
        super().__init__(name="tmz")
        self.client = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "ai-research-agent/1.0"},
        )

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        limit = kwargs.get("limit", 10)
        response = await self.client.get(self.RSS_URL)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        channel = root.find("channel")
        if channel is None:
            return []

        items: list[CollectedItem] = []
        query_lower = query.lower().strip() if query else ""

        for item_el in channel.findall("item"):
            title = item_el.findtext("title", "")
            link = item_el.findtext("link", "")
            pub_date = item_el.findtext("pubDate", "")
            description = item_el.findtext("description", "")

            # Filter by query if provided
            if query_lower:
                if query_lower not in title.lower() and query_lower not in description.lower():
                    continue

            items.append(
                CollectedItem(
                    source="tmz",
                    title=title,
                    content=description[:500],
                    url=link,
                    published_at=pub_date,
                )
            )

            if len(items) >= limit:
                break

        return items

    async def close(self) -> None:
        await self.client.aclose()
