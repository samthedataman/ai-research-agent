import asyncio
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from src.logging_config import get_logger

logger = get_logger(__name__)


class CollectedItem(BaseModel):
    source: str
    title: str
    content: str
    url: str = ""
    published_at: str = ""
    metadata: dict[str, Any] = {}


class BaseCollector(ABC):
    """Base class for data collectors with rate limiting and backoff."""

    def __init__(self, name: str, max_retries: int = 3, base_delay: float = 1.0):
        self.name = name
        self.max_retries = max_retries
        self.base_delay = base_delay

    @abstractmethod
    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """Fetch data for a given query. Override in subclasses."""
        ...

    async def collect(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """Collect data with retry and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                items = await self._fetch(query, **kwargs)
                logger.info(
                    "collection_success",
                    collector=self.name,
                    query=query,
                    items_count=len(items),
                )
                return items
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        "collection_failed",
                        collector=self.name,
                        query=query,
                        error=str(e),
                    )
                    raise
                delay = self.base_delay * (2**attempt)
                logger.warning(
                    "collection_retry",
                    collector=self.name,
                    query=query,
                    attempt=attempt + 1,
                    delay=delay,
                )
                await asyncio.sleep(delay)
        return []
