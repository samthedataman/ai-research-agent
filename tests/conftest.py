import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["TELEGRAM_BOT_TOKEN"] = "test-token"
