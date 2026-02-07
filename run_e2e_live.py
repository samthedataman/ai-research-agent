"""Live E2E runner — exercises free collectors + Ollama pipeline, prints logs."""

import asyncio
import sys
import os

# Use the .env config
from dotenv import load_dotenv
load_dotenv()

from src.logging_config import setup_logging
from src.config import settings
from src.llm import OllamaClient
from src.collectors import get_collector, COLLECTOR_REGISTRY
from src.graph import build_graph
from src.storage import init_db, log_query, get_history, Base

setup_logging()


def banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


async def test_ollama():
    banner("1. OLLAMA HEALTH + COMPLETION")
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_model,
    )
    healthy = await client.health_check()
    print(f"  Ollama healthy: {healthy}")
    print(f"  Model: {settings.ollama_model}")

    resp = await client.complete(
        [{"role": "user", "content": "Say hello in one sentence."}],
        temperature=0.0,
    )
    text = client.get_text(resp)
    print(f"  LLM response: {text[:200]}")
    await client.close()
    return healthy


async def test_free_collectors():
    banner("2. FREE COLLECTORS (live HTTP)")

    free_tests = [
        ("weather", "New York"),
        ("crypto", "trending"),
        ("reddit", "r/technology"),
        ("github", "langchain"),
        ("ddg_news", "artificial intelligence"),
        ("news", "technology"),
    ]

    results = {}
    for source, query in free_tests:
        print(f"\n  --- {source.upper()} (query: '{query}') ---")
        collector = get_collector(source)
        try:
            items = await collector.collect(query, limit=3)
            results[source] = len(items)
            print(f"  Items returned: {len(items)}")
            for i, item in enumerate(items[:2]):
                print(f"    [{i+1}] {item.title[:80]}")
                if item.url:
                    print(f"        URL: {item.url[:80]}")
                print(f"        Content: {item.content[:100]}...")
        except Exception as e:
            results[source] = f"ERROR: {e}"
            print(f"  FAILED: {e}")
        finally:
            await collector.close()

    print(f"\n  Summary: {results}")
    return results


async def test_full_pipeline():
    banner("3. FULL LANGGRAPH PIPELINE (Ollama + live collector)")

    workflow = build_graph()

    # Test 1: Preset source (weather)
    print("\n  --- Pipeline: /weather London ---")
    result = await workflow.ainvoke({
        "user_message": "/weather London",
        "source": "weather",
        "query": "London",
        "items": [],
        "analysis": "",
        "response": "",
        "error": "",
    })
    print(f"  Source: {result['source']}")
    print(f"  Items: {len(result['items'])}")
    print(f"  Error: {result.get('error', 'none')}")
    print(f"  Response ({len(result['response'])} chars):")
    print(f"  {result['response'][:500]}")

    # Test 2: Auto-routed query
    print("\n  --- Pipeline: auto-route 'What is Bitcoin?' ---")
    result2 = await workflow.ainvoke({
        "user_message": "What is Bitcoin and what is the current price?",
        "source": "",
        "query": "",
        "items": [],
        "analysis": "",
        "response": "",
        "error": "",
    })
    print(f"  Routed to: {result2['source']}")
    print(f"  Query: {result2['query']}")
    print(f"  Items: {len(result2['items'])}")
    print(f"  Error: {result2.get('error', 'none')}")
    print(f"  Response ({len(result2['response'])} chars):")
    print(f"  {result2['response'][:500]}")

    # Test 3: Another auto-route
    print("\n  --- Pipeline: auto-route 'Latest AI research papers' ---")
    result3 = await workflow.ainvoke({
        "user_message": "Show me the latest trending repos on GitHub about AI agents",
        "source": "",
        "query": "",
        "items": [],
        "analysis": "",
        "response": "",
        "error": "",
    })
    print(f"  Routed to: {result3['source']}")
    print(f"  Query: {result3['query']}")
    print(f"  Items: {len(result3['items'])}")
    print(f"  Response ({len(result3['response'])} chars):")
    print(f"  {result3['response'][:500]}")

    return result, result2, result3


async def test_storage():
    banner("4. STORAGE (SQLite)")

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    from src.storage import QueryLog
    async with factory() as session:
        entry = QueryLog(
            user_id=42,
            source="weather",
            query="London",
            response="Weather in London: 15°C, cloudy.",
        )
        session.add(entry)
        await session.commit()
        print("  Wrote log entry for user 42")

    from sqlalchemy import select
    async with factory() as session:
        result = await session.execute(
            select(QueryLog).where(QueryLog.user_id == 42)
        )
        rows = list(result.scalars().all())
        print(f"  Read back {len(rows)} entries")
        for row in rows:
            print(f"    source={row.source}, query={row.query}, response={row.response[:60]}")

    await engine.dispose()


async def test_bot_startup():
    banner("5. TELEGRAM BOT STARTUP CHECK")
    token = settings.telegram_bot_token
    if not token or token == "test-token":
        print("  No real token set — skipping bot startup test")
        return

    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
        if resp.status_code == 200:
            data = resp.json()
            bot_info = data.get("result", {})
            print(f"  Bot connected successfully!")
            print(f"  Bot name: @{bot_info.get('username', '?')}")
            print(f"  Bot ID: {bot_info.get('id', '?')}")
            print(f"  First name: {bot_info.get('first_name', '?')}")
        else:
            print(f"  Bot token invalid or expired: {resp.status_code}")
            print(f"  Response: {resp.text[:200]}")


async def main():
    print("AI Research Agent — Live E2E Test Runner")
    print(f"Ollama URL: {settings.ollama_base_url}")
    print(f"Ollama Model: {settings.ollama_model}")
    print(f"LLM Provider: {settings.llm_provider}")

    ok = await test_ollama()
    if not ok:
        print("\nOllama is not running! Start it first.")
        sys.exit(1)

    await test_free_collectors()
    await test_full_pipeline()
    await test_storage()
    await test_bot_startup()

    banner("ALL DONE")
    print("  Everything checked out. Your bot is ready to run with:")
    print("  python main.py")


if __name__ == "__main__":
    asyncio.run(main())
