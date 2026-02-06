"""
Reddit Collector - Public JSON API (FREE, no API key)
=====================================================
Reddit exposes .json endpoints on every page. No OAuth or API key needed.
Rate limit: be respectful, ~1 req/sec is fine.
"""

from typing import Any

import httpx

from src.collectors.base import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class RedditCollector(BaseCollector):
    """Collects posts from Reddit public JSON API (free, no API key)."""

    BASE_URL = "https://www.reddit.com"

    def __init__(self):
        super().__init__(name="reddit")
        self.client = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "ai-research-agent/1.0 (research bot)"},
        )

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """
        Fetch from Reddit. Query can be:
        - A subreddit name like "r/technology" or "technology"
        - A search query (will search all of Reddit)
        """
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", "hot")  # hot, new, top, rising

        if query.startswith("r/") or query.startswith("/r/"):
            return await self._fetch_subreddit(query.lstrip("/").lstrip("r/"), limit, sort)
        else:
            return await self._fetch_search(query, limit, sort)

    async def _fetch_subreddit(
        self, subreddit: str, limit: int, sort: str
    ) -> list[CollectedItem]:
        """Fetch posts from a subreddit."""
        response = await self.client.get(
            f"{self.BASE_URL}/r/{subreddit}/{sort}.json",
            params={"limit": str(limit), "raw_json": "1"},
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_listing(data, f"r/{subreddit}")

    async def _fetch_search(
        self, query: str, limit: int, sort: str
    ) -> list[CollectedItem]:
        """Search Reddit for posts matching query."""
        response = await self.client.get(
            f"{self.BASE_URL}/search.json",
            params={
                "q": query,
                "limit": str(limit),
                "sort": "relevance",
                "t": "week",
                "raw_json": "1",
            },
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_listing(data, f"search:{query}")

    def _parse_listing(self, data: dict, source_label: str) -> list[CollectedItem]:
        """Parse Reddit listing JSON into CollectedItems."""
        items: list[CollectedItem] = []

        children = data.get("data", {}).get("children", [])
        for post_data in children:
            post = post_data.get("data", {})
            if not post:
                continue

            title = post.get("title", "")
            selftext = post.get("selftext", "")[:500]
            subreddit = post.get("subreddit", "")
            score = post.get("score", 0)
            num_comments = post.get("num_comments", 0)
            author = post.get("author", "[deleted]")
            permalink = post.get("permalink", "")
            created_utc = post.get("created_utc", 0)
            url = post.get("url", "")
            is_self = post.get("is_self", False)

            content = f"[r/{subreddit}] {title}"
            if selftext:
                content += f"\n\n{selftext}"
            content += f"\n\nScore: {score} | Comments: {num_comments} | Author: u/{author}"

            items.append(
                CollectedItem(
                    source=f"reddit_{source_label}",
                    title=title,
                    content=content,
                    url=f"https://www.reddit.com{permalink}" if permalink else url,
                    published_at=str(int(created_utc)),
                    metadata={
                        "subreddit": subreddit,
                        "score": score,
                        "num_comments": num_comments,
                        "author": author,
                        "is_self": is_self,
                        "external_url": url if not is_self else "",
                    },
                )
            )
        return items

    async def close(self) -> None:
        await self.client.aclose()
