import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from src.analyzers.sentiment import SentimentAnalyzer
from src.analyzers.topic_extractor import TopicExtractor
from src.analyzers.summarizer import Summarizer


def make_mock_llm(response_text: str):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"mock": True})
    llm.get_text = MagicMock(return_value=response_text)
    return llm


class TestSentimentAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_positive(self):
        result_json = json.dumps(
            {"sentiment": "positive", "confidence": 0.9, "reasoning": "Upbeat tone"}
        )
        llm = make_mock_llm(result_json)
        analyzer = SentimentAnalyzer(llm)

        result = await analyzer.analyze("This is great news!")
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_analyze_with_extra_text(self):
        # LLM sometimes returns JSON wrapped in extra text
        result_json = 'Here is the analysis: {"sentiment": "negative", "confidence": 0.8, "reasoning": "Bad"}'
        llm = make_mock_llm(result_json)
        analyzer = SentimentAnalyzer(llm)

        result = await analyzer.analyze("Bad news today")
        assert result["sentiment"] == "negative"

    @pytest.mark.asyncio
    async def test_analyze_parse_error(self):
        llm = make_mock_llm("This is not valid JSON at all")
        analyzer = SentimentAnalyzer(llm)

        result = await analyzer.analyze("Some text")
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.0


class TestTopicExtractor:
    @pytest.mark.asyncio
    async def test_extract_topics(self):
        result_json = json.dumps(
            {
                "topics": ["AI", "machine learning", "neural networks"],
                "primary_topic": "AI",
                "category": "technology",
            }
        )
        llm = make_mock_llm(result_json)
        extractor = TopicExtractor(llm)

        result = await extractor.extract("Article about AI and ML")
        assert "AI" in result["topics"]
        assert result["category"] == "technology"

    @pytest.mark.asyncio
    async def test_extract_parse_error(self):
        llm = make_mock_llm("invalid json")
        extractor = TopicExtractor(llm)

        result = await extractor.extract("Some text")
        assert result["primary_topic"] == "unknown"


class TestSummarizer:
    @pytest.mark.asyncio
    async def test_summarize(self):
        llm = make_mock_llm("This is a concise summary of the article.")
        summarizer = Summarizer(llm)

        result = await summarizer.summarize("Long article text here...")
        assert "summary" in result.lower()

    @pytest.mark.asyncio
    async def test_create_digest(self):
        llm = make_mock_llm("## Key Items\n- Item 1\n\n## Trend Analysis\nTrends are positive.")
        summarizer = Summarizer(llm)

        result = await summarizer.create_digest(
            [
                {"title": "Title 1", "content": "Content 1"},
                {"title": "Title 2", "content": "Content 2"},
            ]
        )
        assert "Key Items" in result
