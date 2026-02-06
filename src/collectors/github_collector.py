"""
GitHub Collector - Public API (FREE, no API key for basic use)
==============================================================
GitHub REST API v3: 60 req/hour unauthenticated, 5000 req/hour with token.
Works without token for basic trending/search.
"""

from typing import Any

import httpx

from src.collectors.base_collector import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class GitHubCollector(BaseCollector):
    """Collects trending repos and search results from GitHub (free)."""

    API_URL = "https://api.github.com"

    def __init__(self, token: str = ""):
        super().__init__(name="github")
        headers: dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ai-research-agent/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.AsyncClient(timeout=15.0, headers=headers)

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """
        Fetch from GitHub. Query can be:
        - "trending" for trending repos
        - a search query for repository search
        """
        limit = kwargs.get("limit", 10)
        language = kwargs.get("language", "")

        if query.lower() == "trending":
            return await self._fetch_trending(limit, language)
        else:
            return await self._fetch_search(query, limit, language)

    async def _fetch_trending(
        self, limit: int, language: str
    ) -> list[CollectedItem]:
        """Fetch trending repos (repos created in last 7 days with most stars)."""
        from datetime import datetime, timedelta

        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        q = f"created:>{week_ago}"
        if language:
            q += f" language:{language}"

        response = await self.client.get(
            f"{self.API_URL}/search/repositories",
            params={
                "q": q,
                "sort": "stars",
                "order": "desc",
                "per_page": str(limit),
            },
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_repos(data.get("items", []))

    async def _fetch_search(
        self, query: str, limit: int, language: str
    ) -> list[CollectedItem]:
        """Search GitHub repositories."""
        q = query
        if language:
            q += f" language:{language}"

        response = await self.client.get(
            f"{self.API_URL}/search/repositories",
            params={
                "q": q,
                "sort": "stars",
                "order": "desc",
                "per_page": str(limit),
            },
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_repos(data.get("items", []))

    def _parse_repos(self, repos: list[dict]) -> list[CollectedItem]:
        """Parse GitHub repo list into CollectedItems."""
        items: list[CollectedItem] = []
        for repo in repos:
            name = repo.get("full_name", "")
            description = repo.get("description", "") or "No description"
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            language = repo.get("language", "Unknown")
            topics = repo.get("topics", [])
            created = repo.get("created_at", "")
            updated = repo.get("updated_at", "")

            content = (
                f"{name}: {description}\n"
                f"Stars: {stars:,} | Forks: {forks:,} | Language: {language}\n"
                f"Topics: {', '.join(topics[:5]) if topics else 'none'}\n"
                f"Updated: {updated[:10]}"
            )

            items.append(
                CollectedItem(
                    source="github",
                    title=f"{name} ({stars:,} stars)",
                    content=content,
                    url=repo.get("html_url", ""),
                    published_at=created,
                    metadata={
                        "full_name": name,
                        "stars": stars,
                        "forks": forks,
                        "language": language,
                        "topics": topics[:10],
                        "open_issues": repo.get("open_issues_count", 0),
                    },
                )
            )
        return items

    async def close(self) -> None:
        await self.client.aclose()
