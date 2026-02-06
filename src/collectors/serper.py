"""Google search via Serper API (requires SERPER_API_KEY)."""

from typing import Any

import httpx

from src.collectors.base import BaseCollector, CollectedItem
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class SerperCollector(BaseCollector):
    """Google search via Serper.dev — requires SERPER_API_KEY."""

    API_URL = "https://google.serper.dev/search"

    def __init__(self):
        super().__init__(name="serper")
        self.api_key = settings.serper_api_key
        self.client = httpx.AsyncClient(timeout=15.0)

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        if not self.api_key:
            raise ValueError(
                "SERPER_API_KEY is required for Serper search. Set it in .env."
            )

        limit = kwargs.get("limit", 10)
        response = await self.client.post(
            self.API_URL,
            json={"q": query, "num": limit},
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

        items: list[CollectedItem] = []
        for r in data.get("organic", [])[:limit]:
            items.append(
                CollectedItem(
                    source="serper",
                    title=r.get("title", ""),
                    content=r.get("snippet", ""),
                    url=r.get("link", ""),
                    published_at=r.get("date", ""),
                    metadata={"position": r.get("position", 0)},
                )
            )
        return items

    async def close(self) -> None:
        await self.client.aclose()
