from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM Provider
    llm_provider: str = Field(default="ollama", description="ollama or openrouter")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-chat"

    # RapidAPI (optional - free Google News RSS is used if not set)
    rapidapi_key: str = ""

    # GitHub token (optional - unauthenticated gets 60 req/hr)
    github_token: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./agent.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Agent scheduling
    collection_interval_minutes: int = 30
    analysis_interval_minutes: int = 60
    report_interval_hours: int = 24

    # Search
    search_keywords: str = "AI,machine learning,LLM,agents"

    # Data sources to collect from (comma-separated)
    # Available: news,weather,crypto,reddit,github,arxiv,stocks,wikipedia
    active_collectors: str = "news,reddit,arxiv"

    # Collector-specific settings
    weather_locations: str = "New York,San Francisco,London"
    stock_symbols: str = "AAPL,GOOGL,MSFT,NVDA,TSLA"
    reddit_subreddits: str = "technology,machinelearning,artificial"
    crypto_coins: str = "trending"

    # Logging
    log_level: str = "INFO"

    @property
    def keywords_list(self) -> list[str]:
        return [k.strip() for k in self.search_keywords.split(",") if k.strip()]

    @property
    def collectors_list(self) -> list[str]:
        return [c.strip() for c in self.active_collectors.split(",") if c.strip()]

    @property
    def weather_locations_list(self) -> list[str]:
        return [l.strip() for l in self.weather_locations.split(",") if l.strip()]

    @property
    def stock_symbols_list(self) -> list[str]:
        return [s.strip() for s in self.stock_symbols.split(",") if s.strip()]

    @property
    def reddit_subreddits_list(self) -> list[str]:
        return [r.strip() for r in self.reddit_subreddits.split(",") if r.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
