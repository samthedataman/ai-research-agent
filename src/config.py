from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Telegram
    telegram_bot_token: str = Field(default="", description="Telegram Bot API token from @BotFather")

    # LLM Provider
    llm_provider: str = Field(default="ollama", description="ollama or openrouter")

    # Ollama (free, local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # OpenRouter (cloud, cheap)
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-chat"

    # Optional API keys (collectors work without these)
    rapidapi_key: str = ""
    github_token: str = ""
    serper_api_key: str = ""
    brave_api_key: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./bot.db"

    # Data sources (comma-separated list of active collectors)
    # Available: news,weather,crypto,dexscreener,reddit,github,arxiv,stocks,wikipedia,ddg,ddg_news,serper,tmz,cryptonews
    active_collectors: str = "news,reddit,arxiv,crypto,github,weather,stocks,wikipedia,ddg,ddg_news,tmz,cryptonews"

    # Collector-specific defaults
    weather_locations: str = "New York,San Francisco,London"
    stock_symbols: str = "AAPL,GOOGL,MSFT,NVDA,TSLA"
    reddit_subreddits: str = "technology,machinelearning,artificial"
    crypto_coins: str = "trending"

    # Logging
    log_level: str = "INFO"

    @property
    def collectors_list(self) -> list[str]:
        return [c.strip() for c in self.active_collectors.split(",") if c.strip()]

    @property
    def weather_locations_list(self) -> list[str]:
        return [loc.strip() for loc in self.weather_locations.split(",") if loc.strip()]

    @property
    def stock_symbols_list(self) -> list[str]:
        return [s.strip() for s in self.stock_symbols.split(",") if s.strip()]

    @property
    def reddit_subreddits_list(self) -> list[str]:
        return [r.strip() for r in self.reddit_subreddits.split(",") if r.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
