from src.collectors.base_collector import BaseCollector, CollectedItem
from src.collectors.news_collector import NewsCollector, DirectNewsCollector
from src.collectors.weather_collector import WeatherCollector
from src.collectors.crypto_collector import CryptoCollector, DexScreenerCollector
from src.collectors.reddit_collector import RedditCollector
from src.collectors.github_collector import GitHubCollector
from src.collectors.arxiv_collector import ArxivCollector
from src.collectors.stocks_collector import StocksCollector
from src.collectors.wiki_collector import WikipediaCollector

__all__ = [
    "BaseCollector",
    "CollectedItem",
    "NewsCollector",
    "DirectNewsCollector",
    "WeatherCollector",
    "CryptoCollector",
    "DexScreenerCollector",
    "RedditCollector",
    "GitHubCollector",
    "ArxivCollector",
    "StocksCollector",
    "WikipediaCollector",
]

# Registry of all available collectors for easy discovery
COLLECTOR_REGISTRY: dict[str, type[BaseCollector]] = {
    "news": DirectNewsCollector,
    "news_rapidapi": NewsCollector,
    "weather": WeatherCollector,
    "crypto": CryptoCollector,
    "dexscreener": DexScreenerCollector,
    "reddit": RedditCollector,
    "github": GitHubCollector,
    "arxiv": ArxivCollector,
    "stocks": StocksCollector,
    "wikipedia": WikipediaCollector,
}


def get_collector(source: str) -> BaseCollector:
    """Get a collector instance by source name."""
    cls = COLLECTOR_REGISTRY.get(source)
    if cls is None:
        raise ValueError(
            f"Unknown collector: {source}. Available: {list(COLLECTOR_REGISTRY.keys())}"
        )
    return cls()
