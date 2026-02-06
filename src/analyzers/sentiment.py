import json
from typing import Any

from src.clients.llm_factory import LLMClient
from src.logging_config import get_logger

logger = get_logger(__name__)

SENTIMENT_PROMPT = """Analyze the sentiment of the following text. Return a JSON object with:
- "sentiment": one of "positive", "negative", "neutral", "mixed"
- "confidence": float between 0.0 and 1.0
- "reasoning": brief explanation

Text to analyze:
{text}

Return ONLY valid JSON, no other text."""


class SentimentAnalyzer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def analyze(self, text: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a sentiment analysis expert. Always respond with valid JSON."},
            {"role": "user", "content": SENTIMENT_PROMPT.format(text=text[:2000])},
        ]

        response = await self.llm.complete(messages, temperature=0.1)
        raw = self.llm.get_text(response)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw[start:end])
            else:
                result = {"sentiment": "neutral", "confidence": 0.0, "reasoning": "Parse error"}

        logger.info("sentiment_analyzed", sentiment=result.get("sentiment"))
        return result
