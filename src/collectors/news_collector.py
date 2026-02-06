from typing import Any

import httpx

from src.collectors.base_collector import BaseCollector, CollectedItem
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class NewsCollector(BaseCollector):
    """Collects news articles via RapidAPI News API."""

    RAPIDAPI_HOST = "real-time-news-data.p.rapidapi.com"
    BASE_URL = f"https://{RAPIDAPI_HOST}"

    def __init__(self):
        super().__init__(name="news")
        self.client = httpx.AsyncClient(
            headers={
                "x-rapidapi-key": settings.rapidapi_key,
                "x-rapidapi-host": self.RAPIDAPI_HOST,
            },
            timeout=30.0,
        )

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        limit = kwargs.get("limit", 10)
        language = kwargs.get("language", "en")

        response = await self.client.get(
            f"{self.BASE_URL}/search",
            params={
                "query": query,
                "limit": str(limit),
                "lang": language,
            },
        )
        response.raise_for_status()
        data = response.json()

        items: list[CollectedItem] = []
        for article in data.get("data", []):
            items.append(
                CollectedItem(
                    source="news",
                    title=article.get("title", ""),
                    content=article.get("snippet", ""),
                    url=article.get("link", ""),
                    published_at=article.get("published_datetime_utc", ""),
                    metadata={
                        "source_name": article.get("source_name", ""),
                        "source_url": article.get("source_url", ""),
                        "photo_url": article.get("photo_url", ""),
                    },
                )
            )
        return items

    async def close(self) -> None:
        await self.client.aclose()


class DirectNewsCollector(BaseCollector):
    """Fallback news collector using free APIs (no RapidAPI key needed)."""

    def __init__(self):
        super().__init__(name="direct_news")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        # Uses a free RSS-to-JSON proxy for Google News
        response = await self.client.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        )
        response.raise_for_status()

        # Parse RSS XML simply
        text = response.text
        items: list[CollectedItem] = []

        import xml.etree.ElementTree as ET

        root = ET.fromstring(text)
        channel = root.find("channel")
        if channel is None:
            return items

        for item_el in channel.findall("item"):
            title = item_el.findtext("title", "")
            link = item_el.findtext("link", "")
            pub_date = item_el.findtext("pubDate", "")
            description = item_el.findtext("description", "")

            items.append(
                CollectedItem(
                    source="google_news",
                    title=title,
                    content=description,
                    url=link,
                    published_at=pub_date,
                )
            )

        return items[: kwargs.get("limit", 10)]

    async def close(self) -> None:
        await self.client.aclose()
