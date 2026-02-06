"""
Wikipedia Collector - Public API (FREE, no API key)
====================================================
Wikipedia REST API + MediaWiki API. Completely free.
Great for current events, facts, and background research.
"""

from typing import Any

import httpx

from src.collectors.base_collector import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class WikipediaCollector(BaseCollector):
    """Collects content from Wikipedia (free, no API key)."""

    REST_URL = "https://en.wikipedia.org/api/rest_v1"
    MW_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self):
        super().__init__(name="wikipedia")
        self.client = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "ai-research-agent/1.0 (research bot)"},
        )

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """
        Fetch from Wikipedia. Query can be:
        - "current_events" for today's current events
        - "on_this_day" for historical events on this day
        - any search term for article search
        """
        limit = kwargs.get("limit", 5)

        if query.lower() in ("current_events", "current events", "news"):
            return await self._fetch_current_events()
        elif query.lower() in ("on_this_day", "today_in_history"):
            return await self._fetch_on_this_day()
        elif query.lower() in ("featured", "featured_article"):
            return await self._fetch_featured()
        else:
            return await self._fetch_search(query, limit)

    async def _fetch_current_events(self) -> list[CollectedItem]:
        """Fetch Wikipedia current events portal content."""
        from datetime import datetime

        today = datetime.utcnow()
        date_str = today.strftime("%Y_%B_%d")

        # Try the current events page
        response = await self.client.get(
            f"{self.MW_URL}",
            params={
                "action": "parse",
                "page": "Portal:Current_events",
                "prop": "wikitext",
                "format": "json",
                "section": "0",
            },
        )
        response.raise_for_status()
        data = response.json()

        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")

        # Simple extraction: get lines that look like events
        import re
        events = []
        for line in wikitext.split("\n"):
            line = line.strip()
            if line.startswith("*") and len(line) > 10:
                # Clean wiki markup
                clean = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", line)
                clean = re.sub(r"'{2,}", "", clean)
                clean = clean.lstrip("* ").strip()
                if clean and len(clean) > 20:
                    events.append(clean)

        items: list[CollectedItem] = []
        for i, event in enumerate(events[:10]):
            items.append(
                CollectedItem(
                    source="wikipedia_current",
                    title=f"Current Event: {event[:80]}",
                    content=event,
                    url="https://en.wikipedia.org/wiki/Portal:Current_events",
                    metadata={"position": i + 1},
                )
            )
        return items

    async def _fetch_on_this_day(self) -> list[CollectedItem]:
        """Fetch 'on this day' events from Wikipedia REST API."""
        from datetime import datetime

        today = datetime.utcnow()
        month = f"{today.month:02d}"
        day = f"{today.day:02d}"

        response = await self.client.get(
            f"{self.REST_URL}/feed/onthisday/events/{month}/{day}"
        )
        response.raise_for_status()
        data = response.json()

        items: list[CollectedItem] = []
        for event in data.get("events", [])[:10]:
            year = event.get("year", "")
            text = event.get("text", "")
            pages = event.get("pages", [])
            url = pages[0].get("content_urls", {}).get("desktop", {}).get("page", "") if pages else ""

            items.append(
                CollectedItem(
                    source="wikipedia_otd",
                    title=f"{year}: {text[:80]}",
                    content=f"On this day in {year}: {text}",
                    url=url,
                    metadata={"year": year},
                )
            )
        return items

    async def _fetch_featured(self) -> list[CollectedItem]:
        """Fetch today's featured article."""
        from datetime import datetime

        today = datetime.utcnow()
        date_str = f"{today.year}/{today.month:02d}/{today.day:02d}"

        response = await self.client.get(
            f"{self.REST_URL}/feed/featured/{date_str}"
        )
        response.raise_for_status()
        data = response.json()

        items: list[CollectedItem] = []

        # Featured article
        tfa = data.get("tfa", {})
        if tfa:
            items.append(
                CollectedItem(
                    source="wikipedia_featured",
                    title=f"Featured: {tfa.get('title', '')}",
                    content=tfa.get("extract", ""),
                    url=tfa.get("content_urls", {}).get("desktop", {}).get("page", ""),
                )
            )

        # Most read articles
        for article in data.get("mostread", {}).get("articles", [])[:5]:
            items.append(
                CollectedItem(
                    source="wikipedia_mostread",
                    title=f"Most Read: {article.get('title', '')}",
                    content=article.get("extract", "")[:300],
                    url=article.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    metadata={"views": article.get("views", 0)},
                )
            )

        return items

    async def _fetch_search(self, query: str, limit: int) -> list[CollectedItem]:
        """Search Wikipedia articles."""
        response = await self.client.get(
            f"{self.MW_URL}",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
                "srprop": "snippet|titlesnippet|sectionsnippet",
                "format": "json",
            },
        )
        response.raise_for_status()
        data = response.json()

        items: list[CollectedItem] = []
        for result in data.get("query", {}).get("search", []):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            # Clean HTML from snippet
            import re
            snippet = re.sub(r"<[^>]+>", "", snippet)

            # Fetch article summary
            summary = await self._get_summary(title)

            items.append(
                CollectedItem(
                    source="wikipedia",
                    title=title,
                    content=summary or snippet,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    metadata={"word_count": result.get("wordcount", 0)},
                )
            )
        return items

    async def _get_summary(self, title: str) -> str:
        """Get article summary via REST API."""
        try:
            encoded = title.replace(" ", "_")
            response = await self.client.get(
                f"{self.REST_URL}/page/summary/{encoded}"
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("extract", "")[:500]
        except Exception:
            pass
        return ""

    async def close(self) -> None:
        await self.client.aclose()
