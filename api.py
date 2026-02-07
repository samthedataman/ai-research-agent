"""FastAPI layer — REST API + WhatsApp webhooks for the AI Research Agent."""

import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.collectors import COLLECTOR_REGISTRY
from src.config import settings
from src.graph import build_graph, AVAILABLE_SOURCES
from src.logging_config import get_logger, setup_logging
from src.storage import init_db, log_query
from src.whatsapp import handle_incoming_message

logger = get_logger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    logger.info("api_started", provider=settings.llm_provider)
    yield


app = FastAPI(
    title="AI Research Agent API",
    description="Search free data sources and get AI-powered summaries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow = build_graph()


# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    message: str = Field(..., description="Your question or search query", min_length=1)
    source: str = Field(default="", description="Force a specific source (optional)")
    model: str = Field(default="", description="Ollama model for routing (optional)")
    analysis_model: str = Field(default="", description="Ollama model for synthesis (optional)")


class QueryResponse(BaseModel):
    source: str
    query: str
    items: list[dict[str, Any]]
    analysis: str
    response: str
    tried_sources: list[str]
    model_used: str


class SourceInfo(BaseModel):
    name: str
    description: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    """Health check — also tells you the LLM provider."""
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "routing_model": settings.ollama_model,
        "analysis_model": settings.ollama_analysis_model,
    }


@app.get("/sources", response_model=list[SourceInfo])
async def list_sources():
    """List all available data sources."""
    descriptions = {
        "news": "Google News (free)",
        "news_rapidapi": "News via RapidAPI (key required)",
        "weather": "Weather from wttr.in (free)",
        "crypto": "Crypto data from CoinGecko (free)",
        "dexscreener": "DEX token data (free)",
        "reddit": "Reddit posts (free)",
        "github": "GitHub repos (free)",
        "arxiv": "Research papers (free)",
        "stocks": "Stock quotes from Yahoo Finance (free)",
        "wikipedia": "Wikipedia articles (free)",
        "ddg": "DuckDuckGo web search (free)",
        "ddg_news": "DuckDuckGo news search (free)",
        "serper": "Google search via Serper (key required)",
        "tmz": "TMZ celebrity news (free)",
        "cryptonews": "CryptoPanic crypto news (free)",
    }
    return [
        SourceInfo(name=name, description=descriptions.get(name, ""))
        for name in COLLECTOR_REGISTRY
    ]


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run a research query through the full pipeline.

    Send a message and optionally specify a source. The AI will:
    1. Route to the best data source (or use the one you specified)
    2. Collect data from that source (with automatic fallback if it fails)
    3. Synthesize results into a structured briefing

    Examples:
    - {"message": "What is Bitcoin doing today?"}
    - {"message": "AI agents", "source": "github"}
    - {"message": "weather in Tokyo", "source": "weather"}
    """
    state = {
        "user_message": req.message,
        "source": req.source,
        "query": req.message if req.source else "",
        "items": [],
        "analysis": "",
        "response": "",
        "error": "",
        "model": req.model,
        "analysis_model": req.analysis_model,
    }

    try:
        result = await workflow.ainvoke(state)
    except Exception as e:
        logger.error("api_query_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Log to DB
    try:
        await log_query(
            user_id=0,  # API user
            source=result.get("source", "auto"),
            query=req.message,
            response=result.get("response", "")[:2000],
        )
    except Exception:
        pass

    a_model = req.analysis_model or settings.ollama_analysis_model
    return QueryResponse(
        source=result.get("source", ""),
        query=result.get("query", ""),
        items=result.get("items", []),
        analysis=result.get("analysis", ""),
        response=result.get("response", ""),
        tried_sources=result.get("tried_sources", []),
        model_used=a_model,
    )


@app.post("/query/{source}", response_model=QueryResponse)
async def query_source(source: str, req: QueryRequest):
    """Query a specific data source directly.

    Path parameter `source` overrides any source in the body.
    """
    if source not in COLLECTOR_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source: {source}. Available: {list(COLLECTOR_REGISTRY.keys())}",
        )
    req.source = source
    return await query(req)


# ── WhatsApp Webhook ─────────────────────────────────────────────────────────


@app.get("/webhook/whatsapp")
async def whatsapp_verify(request: Request):
    """WhatsApp webhook verification — Meta sends a GET to verify your endpoint."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        logger.info("whatsapp_webhook_verified")
        return Response(content=challenge, media_type="text/plain")

    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """Receive incoming WhatsApp messages and dispatch to the pipeline."""
    raw_body = await request.body()

    # Optional: verify signature from X-Hub-Signature-256 header
    if settings.whatsapp_app_secret:
        signature = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(
            settings.whatsapp_app_secret.encode(),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise HTTPException(status_code=403, detail="Invalid signature")

    body = json.loads(raw_body)

    # Extract messages from the webhook payload
    try:
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for message in value.get("messages", []):
                    if message.get("type") == "text":
                        phone = message["from"]
                        text = message["text"]["body"]
                        # Fire and forget — must respond 200 quickly
                        asyncio.create_task(handle_incoming_message(phone, text))
    except Exception as e:
        logger.error("whatsapp_webhook_parse_error", error=str(e))

    # WhatsApp requires a fast 200 OK
    return {"status": "ok"}


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
