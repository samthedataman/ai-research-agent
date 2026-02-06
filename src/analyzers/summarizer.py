from src.clients.llm_factory import LLMClient
from src.logging_config import get_logger

logger = get_logger(__name__)

SUMMARY_PROMPT = """Summarize the following text in 2-3 concise sentences. Focus on the key insights and findings.

Text to summarize:
{text}

Summary:"""

DIGEST_PROMPT = """Create a brief research digest from the following items. For each item, provide a one-line summary. Then provide an overall trend analysis in 2-3 sentences.

Items:
{items}

Format your response as:
## Key Items
- [item summaries]

## Trend Analysis
[your analysis]"""


class Summarizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def summarize(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a research analyst. Write clear, concise summaries."},
            {"role": "user", "content": SUMMARY_PROMPT.format(text=text[:3000])},
        ]
        response = await self.llm.complete(messages, temperature=0.3)
        summary = self.llm.get_text(response)
        logger.info("text_summarized", length=len(summary))
        return summary

    async def create_digest(self, items: list[dict[str, str]]) -> str:
        items_text = "\n".join(
            f"- Title: {item['title']}\n  Content: {item['content'][:200]}"
            for item in items
        )
        messages = [
            {"role": "system", "content": "You are a research analyst. Create insightful digests."},
            {"role": "user", "content": DIGEST_PROMPT.format(items=items_text[:4000])},
        ]
        response = await self.llm.complete(messages, temperature=0.4)
        digest = self.llm.get_text(response)
        logger.info("digest_created", item_count=len(items))
        return digest
