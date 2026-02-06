import asyncio
import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.analyzers.pipeline import AnalysisPipeline
from src.analyzers.summarizer import Summarizer
from src.clients.llm_factory import get_llm_client
from src.collectors import get_collector, COLLECTOR_REGISTRY
from src.collectors.base_collector import CollectedItem
from src.config import settings
from src.logging_config import get_logger, setup_logging
from src.storage.database import async_session, init_db
from src.storage.repository import Repository

logger = get_logger(__name__)


async def _save_items(items: list[CollectedItem], label: str) -> None:
    """Save collected items to DB and vector store."""
    if not items:
        return

    async with async_session() as session:
        repo = Repository(session)
        ids = await repo.save_collected_items(items)

        # Add to vector store for RAG (non-critical)
        try:
            from src.tools.vector_store import VectorStore

            store = VectorStore()
            docs = [
                {
                    "id": item_id,
                    "text": f"{item.title}\n{item.content}",
                    "metadata": {"source": item.source, "url": item.url},
                }
                for item_id, item in zip(ids, items)
            ]
            store.add_documents(docs)
        except Exception as e:
            logger.warning("vector_store_error", error=str(e))

    logger.info("items_saved", label=label, count=len(items))


async def collect_news() -> None:
    """Collect news articles for all keywords."""
    collector = get_collector("news_rapidapi" if settings.rapidapi_key else "news")
    try:
        for keyword in settings.keywords_list:
            items = await collector.collect(keyword, limit=10)
            await _save_items(items, f"news:{keyword}")
    finally:
        await collector.close()


async def collect_weather() -> None:
    """Collect weather data for configured locations."""
    collector = get_collector("weather")
    try:
        for location in settings.weather_locations_list:
            items = await collector.collect(location)
            await _save_items(items, f"weather:{location}")
    finally:
        await collector.close()


async def collect_crypto() -> None:
    """Collect crypto market data."""
    collector = get_collector("crypto")
    try:
        if settings.crypto_coins.lower() == "trending":
            items = await collector.collect("trending")
        else:
            items = await collector.collect("market", limit=10)
        await _save_items(items, "crypto")
    finally:
        await collector.close()


async def collect_reddit() -> None:
    """Collect posts from configured subreddits."""
    collector = get_collector("reddit")
    try:
        for subreddit in settings.reddit_subreddits_list:
            items = await collector.collect(f"r/{subreddit}", limit=10)
            await _save_items(items, f"reddit:{subreddit}")
    finally:
        await collector.close()


async def collect_github() -> None:
    """Collect trending GitHub repositories."""
    from src.collectors.github_collector import GitHubCollector

    collector = GitHubCollector(token=settings.github_token)
    try:
        items = await collector.collect("trending", limit=10)
        await _save_items(items, "github:trending")
    finally:
        await collector.close()


async def collect_arxiv() -> None:
    """Collect recent arXiv papers for search keywords."""
    collector = get_collector("arxiv")
    try:
        for keyword in settings.keywords_list:
            items = await collector.collect(keyword, limit=5)
            await _save_items(items, f"arxiv:{keyword}")
    finally:
        await collector.close()


async def collect_stocks() -> None:
    """Collect stock market data for configured symbols."""
    collector = get_collector("stocks")
    try:
        symbols = ",".join(settings.stock_symbols_list)
        items = await collector.collect(symbols)
        await _save_items(items, "stocks")
    finally:
        await collector.close()


async def collect_wikipedia() -> None:
    """Collect Wikipedia current events and featured content."""
    collector = get_collector("wikipedia")
    try:
        items = await collector.collect("current_events")
        await _save_items(items, "wikipedia:current")

        featured = await collector.collect("featured")
        await _save_items(featured, "wikipedia:featured")
    finally:
        await collector.close()


# Map collector names to their functions
COLLECTOR_TASKS = {
    "news": collect_news,
    "weather": collect_weather,
    "crypto": collect_crypto,
    "reddit": collect_reddit,
    "github": collect_github,
    "arxiv": collect_arxiv,
    "stocks": collect_stocks,
    "wikipedia": collect_wikipedia,
}


async def collect_task() -> None:
    """Scheduled task: collect data from all active sources."""
    logger.info("collect_task_started", collectors=settings.collectors_list)

    for collector_name in settings.collectors_list:
        task_func = COLLECTOR_TASKS.get(collector_name)
        if task_func is None:
            logger.warning("unknown_collector", name=collector_name)
            continue

        try:
            await task_func()
        except Exception as e:
            logger.error(
                "collector_error",
                collector=collector_name,
                error=str(e),
            )

    logger.info("collect_task_finished")


async def analyze_task() -> None:
    """Scheduled task: analyze pending items."""
    logger.info("analyze_task_started")

    llm = get_llm_client()
    try:
        async with async_session() as session:
            repo = Repository(session)
            pipeline = AnalysisPipeline(llm, repo)
            results = await pipeline.process_pending(limit=50)
    finally:
        await llm.close()

    logger.info("analyze_task_finished", analyzed=len(results))


async def report_task() -> None:
    """Scheduled task: generate daily digest report."""
    logger.info("report_task_started")

    since = datetime.datetime.utcnow() - datetime.timedelta(
        hours=settings.report_interval_hours
    )

    async with async_session() as session:
        repo = Repository(session)
        items = await repo.get_items_since(since)

        if not items:
            logger.info("report_skipped", reason="no items")
            return

        llm = get_llm_client()
        try:
            summarizer = Summarizer(llm)
            items_data = [{"title": i.title, "content": i.content} for i in items]
            digest = await summarizer.create_digest(items_data)

            await repo.save_report(
                report_type="daily",
                title=f"Daily Digest - {datetime.datetime.utcnow().strftime('%Y-%m-%d')}",
                content=digest,
                items_analyzed=[i.id for i in items],
            )
        finally:
            await llm.close()

    logger.info("report_task_finished")


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        collect_task,
        "interval",
        minutes=settings.collection_interval_minutes,
        id="collect",
        name="Data Collection",
    )

    scheduler.add_job(
        analyze_task,
        "interval",
        minutes=settings.analysis_interval_minutes,
        id="analyze",
        name="AI Analysis",
    )

    scheduler.add_job(
        report_task,
        "interval",
        hours=settings.report_interval_hours,
        id="report",
        name="Daily Report",
    )

    return scheduler


async def run_agent() -> None:
    """Run the agent with scheduler and API server."""
    setup_logging()
    await init_db()

    logger.info(
        "agent_starting",
        provider=settings.llm_provider,
        keywords=settings.keywords_list,
        collectors=settings.collectors_list,
    )

    # Run initial collection
    await collect_task()

    scheduler = create_scheduler()
    scheduler.start()

    logger.info("scheduler_started")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("agent_stopped")
