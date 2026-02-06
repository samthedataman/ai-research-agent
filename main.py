"""AI Research Agent — Telegram Bot powered by LangGraph."""

import asyncio

from src.bot import start_bot


if __name__ == "__main__":
    asyncio.run(start_bot())
