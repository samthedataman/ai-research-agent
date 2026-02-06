import datetime
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.clients.llm_factory import get_llm_client
from src.collectors import get_collector, COLLECTOR_REGISTRY
from src.config import settings
from src.logging_config import setup_logging, get_logger
from src.storage.database import async_session, init_db
from src.storage.repository import Repository
from src.analyzers.pipeline import AnalysisPipeline

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    logger.info("app_started", provider=settings.llm_provider)
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="AI Research Agent",
    description="Autonomous AI research agent with LLM-powered analysis and free data sources",
    version="0.2.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---


class CollectRequest(BaseModel):
    query: str
    limit: int = 10
    source: str = "news"


class AnalyzeRequest(BaseModel):
    limit: int = 50


class DigestRequest(BaseModel):
    hours: int = 24


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_healthy: bool
    database: str
    active_collectors: list[str]
    available_collectors: list[str]


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    llm = get_llm_client()
    try:
        llm_ok = await llm.health_check()
    except Exception:
        llm_ok = False
    finally:
        await llm.close()

    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        llm_healthy=llm_ok,
        database=settings.database_url.split("://")[0],
        active_collectors=settings.collectors_list,
        available_collectors=list(COLLECTOR_REGISTRY.keys()),
    )


@app.post("/collect")
async def collect_data(request: CollectRequest) -> dict[str, Any]:
    """Collect data from any available source.

    Sources: news, weather, crypto, dexscreener, reddit, github, arxiv, stocks, wikipedia
    """
    try:
        collector = get_collector(request.source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        items = await collector.collect(request.query, limit=request.limit)
    finally:
        await collector.close()

    async with async_session() as session:
        repo = Repository(session)
        ids = await repo.save_collected_items(items)

    return {
        "source": request.source,
        "query": request.query,
        "collected": len(items),
        "item_ids": ids,
    }


@app.get("/sources")
async def list_sources() -> dict[str, Any]:
    """List all available data sources with descriptions."""
    sources = {
        "news": {
            "description": "Google News RSS (free, no API key)",
            "example_query": "artificial intelligence",
            "requires_key": False,
        },
        "news_rapidapi": {
            "description": "RapidAPI real-time news",
            "example_query": "technology",
            "requires_key": True,
        },
        "weather": {
            "description": "Weather data from wttr.in (free)",
            "example_query": "New York",
            "requires_key": False,
        },
        "crypto": {
            "description": "Crypto market data from CoinGecko (free)",
            "example_query": "trending OR bitcoin OR market",
            "requires_key": False,
        },
        "dexscreener": {
            "description": "DEX token data from DexScreener (free)",
            "example_query": "PEPE",
            "requires_key": False,
        },
        "reddit": {
            "description": "Reddit posts via public JSON API (free)",
            "example_query": "r/technology OR search term",
            "requires_key": False,
        },
        "github": {
            "description": "GitHub repos via public API (free, 60 req/hr)",
            "example_query": "trending OR AI agents",
            "requires_key": False,
        },
        "arxiv": {
            "description": "Research papers from arXiv (free)",
            "example_query": "large language models",
            "requires_key": False,
        },
        "stocks": {
            "description": "Stock quotes from Yahoo Finance (free)",
            "example_query": "AAPL,GOOGL,MSFT OR market",
            "requires_key": False,
        },
        "wikipedia": {
            "description": "Wikipedia content and current events (free)",
            "example_query": "current_events OR search term",
            "requires_key": False,
        },
    }
    return {"sources": sources, "total": len(sources)}


@app.post("/analyze")
async def analyze_pending(request: AnalyzeRequest) -> dict[str, Any]:
    llm = get_llm_client()
    try:
        async with async_session() as session:
            repo = Repository(session)
            pipeline = AnalysisPipeline(llm, repo)
            results = await pipeline.process_pending(limit=request.limit)
    finally:
        await llm.close()

    return {"analyzed": len(results), "results": results}


@app.post("/digest")
async def create_digest(request: DigestRequest) -> dict[str, Any]:
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=request.hours)

    async with async_session() as session:
        repo = Repository(session)
        items = await repo.get_items_since(since)

        if not items:
            raise HTTPException(status_code=404, detail="No items found in the given timeframe")

        llm = get_llm_client()
        try:
            from src.analyzers.summarizer import Summarizer

            summarizer = Summarizer(llm)
            items_data = [{"title": i.title, "content": i.content} for i in items]
            digest = await summarizer.create_digest(items_data)

            report_id = await repo.save_report(
                report_type="digest",
                title=f"Digest - last {request.hours} hours",
                content=digest,
                items_analyzed=[i.id for i in items],
            )
        finally:
            await llm.close()

    return {"report_id": report_id, "items_count": len(items), "digest": digest}


@app.get("/items")
async def list_items(
    limit: int = 50, source: str | None = None, status: str | None = None
) -> dict[str, Any]:
    async with async_session() as session:
        repo = Repository(session)
        since = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        items = await repo.get_items_since(since, source=source)

        results = []
        for item in items[:limit]:
            if status and item.analyzed != status:
                continue
            results.append(
                {
                    "id": item.id,
                    "source": item.source,
                    "title": item.title,
                    "url": item.url,
                    "collected_at": str(item.collected_at),
                    "analyzed": item.analyzed,
                }
            )

    return {"items": results, "total": len(results)}


@app.get("/analyses")
async def list_analyses(
    limit: int = 100, analysis_type: str | None = None
) -> dict[str, Any]:
    async with async_session() as session:
        repo = Repository(session)
        analyses = await repo.get_recent_analyses(limit=limit, analysis_type=analysis_type)

        results = []
        for a in analyses:
            results.append(
                {
                    "id": a.id,
                    "item_id": a.item_id,
                    "type": a.analysis_type,
                    "sentiment": a.sentiment,
                    "confidence": a.confidence,
                    "topics": a.topics,
                    "result": a.result[:500],
                    "model": a.model_used,
                    "created_at": str(a.created_at),
                }
            )

    return {"analyses": results, "total": len(results)}


@app.get("/reports")
async def list_reports(
    limit: int = 10, report_type: str | None = None
) -> dict[str, Any]:
    async with async_session() as session:
        repo = Repository(session)
        reports = await repo.get_reports(report_type=report_type, limit=limit)

        results = []
        for r in reports:
            results.append(
                {
                    "id": r.id,
                    "type": r.report_type,
                    "title": r.title,
                    "content": r.content,
                    "items_analyzed": r.items_analyzed,
                    "created_at": str(r.created_at),
                }
            )

    return {"reports": results, "total": len(results)}
