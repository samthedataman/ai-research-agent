from src.collectors.base import BaseCollector, CollectedItem
from src.collectors.news import NewsCollector, DirectNewsCollector
from src.collectors.weather import WeatherCollector
from src.collectors.crypto import CryptoCollector, DexScreenerCollector
from src.collectors.reddit import RedditCollector
from src.collectors.github import GitHubCollector
from src.collectors.arxiv import ArxivCollector
from src.collectors.stocks import StocksCollector
from src.collectors.wiki import WikipediaCollector
from src.collectors.ddg import DdgWebCollector, DdgNewsCollector
from src.collectors.serper import SerperCollector
from src.collectors.tmz import TmzCollector
from src.collectors.cryptonews import CryptoPanicCollector

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
    "DdgWebCollector",
    "DdgNewsCollector",
    "SerperCollector",
    "TmzCollector",
    "CryptoPanicCollector",
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
    "ddg": DdgWebCollector,
    "ddg_news": DdgNewsCollector,
    "serper": SerperCollector,
    "tmz": TmzCollector,
    "cryptonews": CryptoPanicCollector,
}


def get_collector(source: str) -> BaseCollector:
    """Get a collector instance by source name."""
    cls = COLLECTOR_REGISTRY.get(source)
    if cls is None:
        raise ValueError(
            f"Unknown collector: {source}. Available: {list(COLLECTOR_REGISTRY.keys())}"
        )
    return cls()
