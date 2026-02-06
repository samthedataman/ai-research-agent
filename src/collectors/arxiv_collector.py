"""
arXiv Collector - Public API (FREE, no API key)
================================================
arXiv API: completely free, no authentication.
Rate limit: ~3 req/sec recommended.
Great for AI/ML research papers.
"""

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from src.collectors.base_collector import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)

# arXiv namespaces
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


class ArxivCollector(BaseCollector):
    """Collects research papers from arXiv (free, no API key)."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        super().__init__(name="arxiv")
        self.client = httpx.AsyncClient(timeout=20.0)

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """
        Fetch papers from arXiv. Query examples:
        - "machine learning" (general search)
        - "cat:cs.AI" (category search)
        - "au:hinton" (author search)
        """
        limit = kwargs.get("limit", 10)
        sort_by = kwargs.get("sort_by", "submittedDate")  # relevance, lastUpdatedDate, submittedDate
        sort_order = kwargs.get("sort_order", "descending")

        response = await self.client.get(
            self.BASE_URL,
            params={
                "search_query": f"all:{query}",
                "start": "0",
                "max_results": str(limit),
                "sortBy": sort_by,
                "sortOrder": sort_order,
            },
        )
        response.raise_for_status()

        return self._parse_atom(response.text)

    def _parse_atom(self, xml_text: str) -> list[CollectedItem]:
        """Parse arXiv Atom XML response."""
        root = ET.fromstring(xml_text)
        items: list[CollectedItem] = []

        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            title_el = entry.find(f"{{{ATOM_NS}}}title")
            summary_el = entry.find(f"{{{ATOM_NS}}}summary")
            published_el = entry.find(f"{{{ATOM_NS}}}published")
            updated_el = entry.find(f"{{{ATOM_NS}}}updated")

            title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
            summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
            published = published_el.text if published_el is not None else ""
            updated = updated_el.text if updated_el is not None else ""

            # Get PDF and abstract URLs
            pdf_url = ""
            abs_url = ""
            for link in entry.findall(f"{{{ATOM_NS}}}link"):
                href = link.get("href", "")
                if link.get("title") == "pdf":
                    pdf_url = href
                elif link.get("type") == "text/html":
                    abs_url = href
                elif not abs_url and "/abs/" in href:
                    abs_url = href

            # Get authors
            authors = []
            for author in entry.findall(f"{{{ATOM_NS}}}author"):
                name_el = author.find(f"{{{ATOM_NS}}}name")
                if name_el is not None:
                    authors.append(name_el.text)

            # Get categories
            categories = []
            for cat in entry.findall(f"{{{ATOM_NS}}}category"):
                term = cat.get("term", "")
                if term:
                    categories.append(term)

            # Get arXiv ID
            id_el = entry.find(f"{{{ATOM_NS}}}id")
            arxiv_id = ""
            if id_el is not None and id_el.text:
                arxiv_id = id_el.text.split("/abs/")[-1]

            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al. ({len(authors)} authors)"

            content = (
                f"Title: {title}\n"
                f"Authors: {author_str}\n"
                f"Abstract: {summary[:500]}\n"
                f"Categories: {', '.join(categories[:5])}\n"
                f"Published: {published[:10]}"
            )

            items.append(
                CollectedItem(
                    source="arxiv",
                    title=title,
                    content=content,
                    url=abs_url or pdf_url,
                    published_at=published,
                    metadata={
                        "arxiv_id": arxiv_id,
                        "authors": authors[:10],
                        "categories": categories,
                        "pdf_url": pdf_url,
                        "updated": updated,
                    },
                )
            )
        return items

    async def close(self) -> None:
        await self.client.aclose()
