"""AI Research Agent — Telegram Bot + Daily Scheduler powered by LangGraph."""

import asyncio

from src.bot import start_bot
from src.config import settings
from src.logging_config import get_logger, setup_logging
from src.storage import init_db

logger = get_logger(__name__)


async def main():
    setup_logging()
    await init_db()

    tasks = [start_bot()]

    # Start daily briefing scheduler if WhatsApp/Green API is configured
    if settings.whatsapp_access_token or settings.greenapi_instance_id:
        from src.scheduler import scheduler_loop
        tasks.append(scheduler_loop())
        logger.info("scheduler_enabled", hour=settings.daily_briefing_hour, group=settings.greenapi_group_id or "none")
    else:
        logger.info("scheduler_disabled", reason="no WhatsApp or Green API configured")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
