import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.collectors.base import CollectedItem
from src.collectors import get_collector, COLLECTOR_REGISTRY


class TestCollectorRegistry:
    def test_all_collectors_registered(self):
        expected = [
            "news", "news_rapidapi", "weather", "crypto", "dexscreener",
            "reddit", "github", "arxiv", "stocks", "wikipedia",
        ]
        for name in expected:
            assert name in COLLECTOR_REGISTRY, f"Missing collector: {name}"

    def test_get_collector_valid(self):
        collector = get_collector("news")
        assert collector is not None
        assert collector.name in ("news", "direct_news")

    def test_get_collector_invalid(self):
        with pytest.raises(ValueError, match="Unknown collector"):
            get_collector("nonexistent")


class TestCollectedItem:
    def test_create_minimal(self):
        item = CollectedItem(source="test", title="Test", content="Content")
        assert item.source == "test"
        assert item.url == ""
        assert item.metadata == {}

    def test_create_full(self):
        item = CollectedItem(
            source="test",
            title="Test Title",
            content="Test content",
            url="https://example.com",
            published_at="2024-01-01",
            metadata={"key": "value"},
        )
        assert item.url == "https://example.com"
        assert item.metadata["key"] == "value"


class TestWeatherCollector:
    @pytest.mark.asyncio
    async def test_fetch_json(self):
        from src.collectors.weather import WeatherCollector

        collector = WeatherCollector()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "current_condition": [
                {
                    "temp_C": "22",
                    "temp_F": "72",
                    "humidity": "55",
                    "windspeedKmph": "10",
                    "FeelsLikeC": "20",
                    "weatherDesc": [{"value": "Sunny"}],
                }
            ],
            "nearest_area": [
                {
                    "areaName": [{"value": "New York"}],
                    "country": [{"value": "USA"}],
                }
            ],
            "weather": [],
        }
        collector.client = AsyncMock()
        collector.client.get = AsyncMock(return_value=mock_response)

        items = await collector._fetch("New York")
        assert len(items) == 1
        assert "22°C" in items[0].content
        assert items[0].source == "weather_wttr"


class TestCryptoCollector:
    @pytest.mark.asyncio
    async def test_fetch_trending(self):
        from src.collectors.crypto import CryptoCollector

        collector = CryptoCollector()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "coins": [
                {
                    "item": {
                        "id": "bitcoin",
                        "name": "Bitcoin",
                        "symbol": "BTC",
                        "market_cap_rank": 1,
                        "price_btc": 1.0,
                    }
                }
            ]
        }
        collector.client = AsyncMock()
        collector.client.get = AsyncMock(return_value=mock_response)

        items = await collector._fetch("trending")
        assert len(items) == 1
        assert "Bitcoin" in items[0].title


class TestRedditCollector:
    @pytest.mark.asyncio
    async def test_fetch_subreddit(self):
        from src.collectors.reddit import RedditCollector

        collector = RedditCollector()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Test Post",
                            "selftext": "Post content",
                            "subreddit": "technology",
                            "score": 100,
                            "num_comments": 50,
                            "author": "testuser",
                            "permalink": "/r/technology/test",
                            "created_utc": 1700000000,
                            "url": "https://reddit.com/r/technology/test",
                            "is_self": True,
                        }
                    }
                ]
            }
        }
        collector.client = AsyncMock()
        collector.client.get = AsyncMock(return_value=mock_response)

        items = await collector._fetch("r/technology")
        assert len(items) == 1
        assert "Test Post" in items[0].title
        assert items[0].metadata["score"] == 100


class TestGitHubCollector:
    @pytest.mark.asyncio
    async def test_fetch_search(self):
        from src.collectors.github import GitHubCollector

        collector = GitHubCollector()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {
                    "full_name": "user/repo",
                    "description": "Test repo",
                    "stargazers_count": 500,
                    "forks_count": 50,
                    "language": "Python",
                    "topics": ["ai", "ml"],
                    "created_at": "2024-01-01",
                    "updated_at": "2024-06-01",
                    "html_url": "https://github.com/user/repo",
                    "open_issues_count": 10,
                }
            ]
        }
        collector.client = AsyncMock()
        collector.client.get = AsyncMock(return_value=mock_response)

        items = await collector._fetch("AI agents")
        assert len(items) == 1
        assert "500" in items[0].title
        assert items[0].metadata["language"] == "Python"


class TestArxivCollector:
    @pytest.mark.asyncio
    async def test_parse_atom(self):
        from src.collectors.arxiv import ArxivCollector

        collector = ArxivCollector()
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345</id>
    <title>Test Paper Title</title>
    <summary>This is the abstract of the paper.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <updated>2024-01-16T00:00:00Z</updated>
    <author><name>John Doe</name></author>
    <author><name>Jane Smith</name></author>
    <link href="http://arxiv.org/abs/2401.12345" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.12345" title="pdf"/>
    <category term="cs.AI"/>
  </entry>
</feed>"""
        items = collector._parse_atom(xml)
        assert len(items) == 1
        assert "Test Paper Title" in items[0].title
        assert "John Doe" in items[0].content
        assert items[0].metadata["arxiv_id"] == "2401.12345"


class TestStocksCollector:
    @pytest.mark.asyncio
    async def test_fetch_quotes_fallback(self):
        from src.collectors.stocks import StocksCollector

        collector = StocksCollector()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "regularMarketPrice": 150.0,
                            "chartPreviousClose": 148.0,
                            "shortName": "Apple Inc.",
                        }
                    }
                ]
            }
        }
        collector.client = AsyncMock()
        collector.client.get = AsyncMock(return_value=mock_response)

        items = await collector._fetch_quotes_fallback(["AAPL"])
        assert len(items) == 1
        assert "AAPL" in items[0].title
