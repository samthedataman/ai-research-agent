import json
from typing import Any

from src.clients.llm_factory import LLMClient
from src.logging_config import get_logger

logger = get_logger(__name__)

TOPIC_PROMPT = """Extract the main topics from the following text. Return a JSON object with:
- "topics": list of 3-5 topic strings (short phrases)
- "primary_topic": the single most important topic
- "category": one of "technology", "business", "science", "politics", "health", "other"

Text to analyze:
{text}

Return ONLY valid JSON, no other text."""


class TopicExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def extract(self, text: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a topic extraction expert. Always respond with valid JSON."},
            {"role": "user", "content": TOPIC_PROMPT.format(text=text[:2000])},
        ]

        response = await self.llm.complete(messages, temperature=0.1)
        raw = self.llm.get_text(response)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw[start:end])
            else:
                result = {"topics": [], "primary_topic": "unknown", "category": "other"}

        logger.info("topics_extracted", topics=result.get("topics"))
        return result
