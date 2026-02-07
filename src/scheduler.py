"""Scheduler for daily WhatsApp briefings — sends to group + individual subscribers."""

import asyncio
from datetime import datetime, timedelta

from src.config import settings
from src.graph import build_graph
from src.logging_config import get_logger
from src.storage import get_wa_subscribers
from src.whatsapp import send_to_group, send_whatsapp_message

logger = get_logger(__name__)

workflow = build_graph()


async def generate_daily_briefing() -> str:
    """Run the pipeline for each daily source and combine into one briefing."""
    sources = settings.daily_briefing_sources_list
    query_map = {
        "news": "top technology and AI news today",
        "crypto": "trending",
        "stocks": settings.stock_symbols,
        "weather": "New York",
        "reddit": "trending technology",
        "arxiv": "latest AI research",
        "github": "trending repositories",
    }

    sections: list[str] = []
    for source in sources:
        state = {
            "user_message": f"Daily {source} update",
            "source": source,
            "query": query_map.get(source, "trending"),
            "items": [],
            "analysis": "",
            "response": "",
            "error": "",
            "model": "",
            "analysis_model": "",
        }
        try:
            result = await workflow.ainvoke(state)
            analysis = result.get("analysis", "")
            if analysis:
                sections.append(f"--- {source.upper()} ---\n{analysis}")
        except Exception as e:
            logger.error("daily_briefing_source_error", source=source, error=str(e))
            sections.append(f"--- {source.upper()} ---\nUnavailable today.")

    header = f"Good morning! Here's your daily briefing for {datetime.utcnow().strftime('%B %d, %Y')}:\n\n"
    return header + "\n\n".join(sections)


async def send_daily_updates() -> None:
    """Generate briefing and send to WhatsApp group + individual subscribers."""
    briefing = await generate_daily_briefing()
    logger.info("daily_briefing_generated", length=len(briefing))

    # Send to WhatsApp group (via Green API)
    if settings.greenapi_group_id:
        try:
            await send_to_group(briefing[:4000])
            logger.info("daily_briefing_sent_to_group", group=settings.greenapi_group_id)
        except Exception as e:
            logger.error("daily_group_send_error", error=str(e))

    # Also send to individual subscribers (DMs)
    subscribers = await get_wa_subscribers()
    if subscribers:
        for sub in subscribers:
            try:
                await send_whatsapp_message(sub.phone_number, briefing[:4096])
            except Exception as e:
                logger.error("daily_send_error", phone=sub.phone_number, error=str(e))
    elif not settings.greenapi_group_id:
        logger.info("no_recipients_for_daily_briefing")


async def scheduler_loop() -> None:
    """Simple async scheduler — runs daily at a configurable hour (UTC)."""
    target_hour = settings.daily_briefing_hour
    target_minute = settings.daily_briefing_minute

    logger.info(
        "scheduler_started",
        target_time=f"{target_hour:02d}:{target_minute:02d} UTC",
        sources=settings.daily_briefing_sources,
        group=settings.greenapi_group_id or "none",
    )

    while True:
        now = datetime.utcnow()
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

        # If we're already past today's target, schedule for tomorrow
        if now >= target:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info("scheduler_waiting", next_run=target.isoformat(), wait_seconds=int(wait_seconds))
        await asyncio.sleep(wait_seconds)

        logger.info("scheduler_triggered")
        try:
            await send_daily_updates()
        except Exception as e:
            logger.error("scheduler_error", error=str(e))

        # Small delay to avoid double-firing
        await asyncio.sleep(60)
