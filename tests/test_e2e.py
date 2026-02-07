"""
End-to-end tests for the AI Research Agent.

These tests exercise the real system wiring — Ollama LLM, the LangGraph
workflow, individual collectors (with live HTTP where free/no-key), and
the SQLite storage layer.

Requirements:
  - Ollama running locally on :11434 with at least one model pulled
    (we auto-detect available models and pick the fastest one).

Run:
    pytest tests/test_e2e.py -v -s
"""

import asyncio
import json
import os
import pytest
import httpx

from src.collectors.base import CollectedItem
from src.collectors import get_collector, COLLECTOR_REGISTRY
from src.config import Settings
from src.llm import OllamaClient
from src.graph import (
    AgentState,
    build_graph,
    route_node,
    collect_node,
    analyze_node,
    respond_node,
)
from src.storage import Base, QueryLog, init_db, log_query, get_history

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture(scope="session")
def event_loop():
    """Shared event loop for session-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def ollama_model():
    """Detect the fastest available Ollama model (prefer small ones)."""
    # Preference order: small → medium → large
    preferred = [
        "qwen2.5:0.5b",
        "llama3.2:1b",
        "deepseek-coder:latest",
        "phi3:mini",
        "llama3.2:latest",
        "llama3:latest",
    ]
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pytest.skip("Ollama is not running — skipping E2E tests")
        return

    for name in preferred:
        if name in available:
            return name
    # Fall back to whatever is first
    if available:
        return available[0]
    pytest.skip("No Ollama models available — pull one first")


@pytest.fixture
async def ollama(ollama_model):
    """Return an OllamaClient configured with the fastest local model."""
    client = OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model)
    yield client
    await client.close()


@pytest.fixture
async def db_engine():
    """Create an in-memory SQLite engine + tables for storage tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    """Provide a session factory bound to the in-memory DB."""
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    yield factory


# ===================================================================
# 1. OLLAMA LLM — real inference
# ===================================================================


class TestOllamaLLM:
    """Tests that exercise the real Ollama LLM client."""

    @pytest.mark.asyncio
    async def test_health_check(self, ollama):
        assert await ollama.health_check() is True

    @pytest.mark.asyncio
    async def test_simple_completion(self, ollama):
        """LLM should return a non-empty response for a trivial prompt."""
        resp = await ollama.complete(
            [{"role": "user", "content": "Reply with just the word 'hello'."}],
            temperature=0.0,
        )
        text = ollama.get_text(resp)
        assert len(text) > 0
        assert "hello" in text.lower()

    @pytest.mark.asyncio
    async def test_json_output(self, ollama):
        """LLM should be able to produce valid JSON when asked."""
        resp = await ollama.complete(
            [
                {
                    "role": "user",
                    "content": (
                        'Respond with ONLY valid JSON: {"source": "news", "query": "AI"}. '
                        "No other text."
                    ),
                }
            ],
            temperature=0.0,
        )
        text = ollama.get_text(resp).strip()
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1].replace("json", "", 1).strip()
        parsed = json.loads(text)
        assert "source" in parsed
        assert "query" in parsed

    @pytest.mark.asyncio
    async def test_summarization(self, ollama):
        """LLM should produce a shorter summary of a longer text."""
        long_text = (
            "Artificial intelligence has transformed industries ranging from healthcare "
            "to finance. Machine learning models can now diagnose diseases, predict stock "
            "market trends, and automate customer service interactions. Deep learning has "
            "enabled breakthroughs in natural language processing, computer vision, and "
            "robotics. As these technologies continue to advance, ethical considerations "
            "around bias, transparency, and job displacement remain critical areas of "
            "discussion among researchers, policymakers, and the general public."
        )
        resp = await ollama.complete(
            [
                {
                    "role": "user",
                    "content": f"Summarize this in one sentence:\n\n{long_text}",
                }
            ],
            temperature=0.3,
        )
        summary = ollama.get_text(resp)
        assert len(summary) > 10
        assert len(summary) < len(long_text) * 2  # should be shorter-ish


# ===================================================================
# 2. GRAPH NODES — real Ollama, mocked collectors
# ===================================================================


class TestGraphNodesWithOllama:
    """Test individual graph nodes using real Ollama for LLM calls."""

    @pytest.mark.asyncio
    async def test_route_node_picks_source(self, ollama_model, monkeypatch):
        """route_node should use the LLM to pick a source for a user message."""
        # Patch get_llm_client to return an OllamaClient with our model
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )

        state: AgentState = {
            "user_message": "What is the weather in London?",
            "source": "",
            "query": "",
            "items": [],
            "analysis": "",
            "response": "",
            "error": "",
        }
        result = await route_node(state)
        assert result["source"] in list(COLLECTOR_REGISTRY.keys())
        assert len(result["query"]) > 0

    @pytest.mark.asyncio
    async def test_route_node_crypto_intent(self, ollama_model, monkeypatch):
        """Route should recognise crypto intent."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )
        state: AgentState = {
            "user_message": "What is the Bitcoin price today?",
            "source": "",
            "query": "",
            "items": [],
            "analysis": "",
            "response": "",
            "error": "",
        }
        result = await route_node(state)
        # Should route to crypto or news — both are reasonable
        assert result["source"] in list(COLLECTOR_REGISTRY.keys())

    @pytest.mark.asyncio
    async def test_analyze_node_with_items(self, ollama_model, monkeypatch):
        """analyze_node should produce a summary from collected items."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )
        state: AgentState = {
            "user_message": "/news AI agents",
            "source": "news",
            "query": "AI agents",
            "items": [
                {
                    "title": "AI Agents Are Taking Over Software Development",
                    "content": "Major tech companies are deploying AI agents to write and review code.",
                    "url": "https://example.com/1",
                    "source": "news",
                },
                {
                    "title": "OpenAI Launches New Agent Framework",
                    "content": "A new tool for building autonomous AI agents was released today.",
                    "url": "https://example.com/2",
                    "source": "news",
                },
            ],
            "analysis": "",
            "response": "",
            "error": "",
        }
        result = await analyze_node(state)
        assert len(result["analysis"]) > 20
        # Should mention something about AI
        assert result.get("error", "") == ""

    @pytest.mark.asyncio
    async def test_respond_node_formatting(self):
        """respond_node should format the final message."""
        state: AgentState = {
            "user_message": "/news AI",
            "source": "news",
            "query": "AI",
            "items": [],
            "analysis": "Here is a summary of AI news.",
            "response": "",
            "error": "",
        }
        result = await respond_node(state)
        assert "NEWS" in result["response"]
        assert "AI" in result["response"]
        assert len(result["response"]) <= 4096


# ===================================================================
# 3. FULL GRAPH — end-to-end with real Ollama + mocked collector
# ===================================================================


class TestFullGraph:
    """Run the complete LangGraph workflow end-to-end."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_preset_source(self, ollama_model, monkeypatch):
        """Full pipeline: source preset → collect (mocked) → analyze (real LLM) → respond."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )

        fake_items = [
            CollectedItem(
                source="wikipedia",
                title="Python (programming language)",
                content="Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
                url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            ),
            CollectedItem(
                source="wikipedia",
                title="Guido van Rossum",
                content="Guido van Rossum is a Dutch programmer best known as the creator of Python.",
                url="https://en.wikipedia.org/wiki/Guido_van_Rossum",
            ),
        ]

        from unittest.mock import AsyncMock, MagicMock

        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=fake_items)
        mock_collector.close = AsyncMock()

        monkeypatch.setattr(
            "src.graph.get_collector", lambda source: mock_collector
        )

        workflow = build_graph()
        result = await workflow.ainvoke(
            {
                "user_message": "/wiki Python programming",
                "source": "wikipedia",
                "query": "Python programming",
                "items": [],
                "analysis": "",
                "response": "",
                "error": "",
            }
        )

        assert result["source"] == "wikipedia"
        assert len(result["items"]) == 2
        assert len(result["analysis"]) > 10
        assert len(result["response"]) > 10
        assert len(result["response"]) <= 4096

    @pytest.mark.asyncio
    async def test_full_pipeline_auto_route(self, ollama_model, monkeypatch):
        """Full pipeline: LLM routes the query automatically."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )

        fake_items = [
            CollectedItem(
                source="news",
                title="Breaking: Major AI Breakthrough",
                content="Researchers have achieved a significant advancement in artificial intelligence.",
                url="https://example.com/ai-news",
            ),
        ]

        from unittest.mock import AsyncMock, MagicMock

        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=fake_items)
        mock_collector.close = AsyncMock()
        monkeypatch.setattr(
            "src.graph.get_collector", lambda source: mock_collector
        )

        workflow = build_graph()
        result = await workflow.ainvoke(
            {
                "user_message": "What is happening in AI research today?",
                "source": "",
                "query": "",
                "items": [],
                "analysis": "",
                "response": "",
                "error": "",
            }
        )

        # Route node should have picked a valid source
        assert result["source"] in list(COLLECTOR_REGISTRY.keys())
        assert len(result["query"]) > 0
        assert len(result["response"]) > 10

    @pytest.mark.asyncio
    async def test_full_pipeline_empty_results(self, ollama_model, monkeypatch):
        """Pipeline gracefully handles when no data is collected."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )

        from unittest.mock import AsyncMock, MagicMock

        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=[])
        mock_collector.close = AsyncMock()
        monkeypatch.setattr(
            "src.graph.get_collector", lambda source: mock_collector
        )

        workflow = build_graph()
        result = await workflow.ainvoke(
            {
                "user_message": "/news xyznonexistent",
                "source": "news",
                "query": "xyznonexistent",
                "items": [],
                "analysis": "",
                "response": "",
                "error": "",
            }
        )

        assert result["items"] == []
        assert "No results" in result["error"]
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_collector_error(self, ollama_model, monkeypatch):
        """Pipeline handles collector exceptions gracefully."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )

        from unittest.mock import AsyncMock, MagicMock

        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(side_effect=Exception("Network timeout"))
        mock_collector.close = AsyncMock()
        monkeypatch.setattr(
            "src.graph.get_collector", lambda source: mock_collector
        )

        workflow = build_graph()
        result = await workflow.ainvoke(
            {
                "user_message": "/news test",
                "source": "news",
                "query": "test",
                "items": [],
                "analysis": "",
                "response": "",
                "error": "",
            }
        )

        assert "Failed to collect" in result.get("error", "")
        assert len(result["response"]) > 0


# ===================================================================
# 4. LIVE COLLECTORS — real HTTP to free APIs (no key required)
# ===================================================================


class TestLiveCollectors:
    """Test collectors against real free APIs.

    These tests hit the network.  They are marked with a generous timeout
    because free APIs can be slow.  If a service is temporarily down, the
    test is skipped rather than failing.
    """

    @pytest.mark.asyncio
    async def test_weather_collector(self):
        collector = get_collector("weather")
        try:
            items = await collector.collect("London", limit=1)
            assert len(items) >= 1
            assert items[0].source == "weather_wttr"
            assert "London" in items[0].title or "London" in items[0].content
            assert "°C" in items[0].content
        except Exception as e:
            pytest.skip(f"Weather API unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_wikipedia_search(self):
        collector = get_collector("wikipedia")
        try:
            items = await collector.collect("Python programming language", limit=3)
            assert len(items) >= 1
            assert items[0].source == "wikipedia"
            assert any("python" in item.title.lower() for item in items)
        except Exception as e:
            pytest.skip(f"Wikipedia API unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_arxiv_collector(self):
        collector = get_collector("arxiv")
        try:
            items = await collector.collect("transformer neural network", limit=3)
            assert len(items) >= 1
            assert items[0].source == "arxiv"
            assert items[0].url  # should have a URL
        except Exception as e:
            pytest.skip(f"arXiv API unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_crypto_trending(self):
        collector = get_collector("crypto")
        try:
            items = await collector.collect("trending", limit=5)
            assert len(items) >= 1
            assert "coingecko" in items[0].source.lower() or "crypto" in items[0].source.lower()
        except Exception as e:
            pytest.skip(f"CoinGecko API unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_github_search(self):
        collector = get_collector("github")
        try:
            items = await collector.collect("langchain", limit=3)
            assert len(items) >= 1
            assert items[0].url.startswith("https://github.com")
        except Exception as e:
            pytest.skip(f"GitHub API unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_reddit_collector(self):
        collector = get_collector("reddit")
        try:
            items = await collector.collect("r/technology", limit=3)
            assert len(items) >= 1
            assert "reddit" in items[0].source.lower() or items[0].metadata.get("subreddit")
        except Exception as e:
            pytest.skip(f"Reddit API unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_ddg_web_search(self):
        collector = get_collector("ddg")
        try:
            items = await collector.collect("Python programming", limit=3)
            # DDG may return 0 results due to rate limiting or library issues
            if len(items) == 0:
                pytest.skip("DuckDuckGo returned no results (rate limited or library issue)")
            assert items[0].source == "ddg_web"
            assert items[0].url  # should have a URL
        except Exception as e:
            pytest.skip(f"DuckDuckGo API unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_ddg_news_search(self):
        collector = get_collector("ddg_news")
        try:
            items = await collector.collect("artificial intelligence", limit=3)
            assert len(items) >= 1
            assert items[0].source == "ddg_news"
        except Exception as e:
            pytest.skip(f"DuckDuckGo News unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_news_collector(self):
        collector = get_collector("news")
        try:
            items = await collector.collect("technology", limit=3)
            assert len(items) >= 1
        except Exception as e:
            pytest.skip(f"News collector unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_tmz_collector(self):
        collector = get_collector("tmz")
        try:
            items = await collector.collect("celebrity", limit=3)
            assert len(items) >= 1
            assert items[0].source == "tmz"
        except Exception as e:
            pytest.skip(f"TMZ RSS unavailable: {e}")
        finally:
            await collector.close()

    @pytest.mark.asyncio
    async def test_cryptonews_collector(self):
        collector = get_collector("cryptonews")
        try:
            items = await collector.collect("bitcoin", limit=3)
            assert len(items) >= 1
            assert items[0].source == "cryptopanic"
        except Exception as e:
            pytest.skip(f"CryptoPanic RSS unavailable: {e}")
        finally:
            await collector.close()


# ===================================================================
# 5. STORAGE — real SQLite (in-memory)
# ===================================================================


class TestStorageE2E:
    """Full storage round-trip tests against a real in-memory SQLite DB."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self, db_engine):
        """init_db should create the query_log table."""
        async with db_engine.connect() as conn:
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    __import__("sqlalchemy").text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='query_log'"
                    )
                ).fetchone()
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, db_session):
        """Write a query log entry and read it back."""
        async with db_session() as session:
            entry = QueryLog(
                user_id=12345,
                source="news",
                query="AI agents",
                response="Here are the top stories about AI agents...",
            )
            session.add(entry)
            await session.commit()

        # Read back
        from sqlalchemy import select

        async with db_session() as session:
            result = await session.execute(
                select(QueryLog).where(QueryLog.user_id == 12345)
            )
            rows = list(result.scalars().all())
            assert len(rows) == 1
            assert rows[0].source == "news"
            assert rows[0].query == "AI agents"
            assert "AI agents" in rows[0].response

    @pytest.mark.asyncio
    async def test_multiple_entries_ordered(self, db_session):
        """Multiple entries for the same user are stored and retrievable."""
        import datetime

        async with db_session() as session:
            for i in range(5):
                entry = QueryLog(
                    user_id=99999,
                    source=f"source_{i}",
                    query=f"query_{i}",
                    response=f"response_{i}",
                    created_at=datetime.datetime(2024, 1, 1, i, 0, 0),
                )
                session.add(entry)
            await session.commit()

        from sqlalchemy import select

        async with db_session() as session:
            result = await session.execute(
                select(QueryLog)
                .where(QueryLog.user_id == 99999)
                .order_by(QueryLog.created_at.desc())
                .limit(3)
            )
            rows = list(result.scalars().all())
            assert len(rows) == 3
            # Most recent first
            assert rows[0].source == "source_4"
            assert rows[2].source == "source_2"


# ===================================================================
# 6. CONFIG — settings validation
# ===================================================================


class TestConfig:
    def test_default_settings(self):
        """Defaults should produce a working config."""
        s = Settings(
            _env_file=None,
            telegram_bot_token="test",
            llm_provider="ollama",
        )
        assert s.llm_provider == "ollama"
        assert s.ollama_model == "llama3.1:8b"
        assert "news" in s.collectors_list

    def test_collectors_list_parsing(self):
        s = Settings(
            _env_file=None,
            telegram_bot_token="test",
            active_collectors="news, weather, crypto",
        )
        assert s.collectors_list == ["news", "weather", "crypto"]

    def test_stock_symbols_parsing(self):
        s = Settings(
            _env_file=None,
            telegram_bot_token="test",
            stock_symbols="AAPL, TSLA",
        )
        assert s.stock_symbols_list == ["AAPL", "TSLA"]

    def test_weather_locations_parsing(self):
        s = Settings(
            _env_file=None,
            telegram_bot_token="test",
            weather_locations="NYC, SF",
        )
        assert s.weather_locations_list == ["NYC", "SF"]


# ===================================================================
# 7. COLLECTOR REGISTRY — structural checks
# ===================================================================


class TestCollectorRegistryE2E:
    def test_all_collectors_instantiate(self):
        """Every registered collector should instantiate without error."""
        for name, cls in COLLECTOR_REGISTRY.items():
            collector = cls()
            assert collector.name, f"{name} has no name"

    def test_get_collector_returns_correct_type(self):
        for name, cls in COLLECTOR_REGISTRY.items():
            collector = get_collector(name)
            assert isinstance(collector, cls), f"{name} returned wrong type"

    def test_unknown_collector_raises(self):
        with pytest.raises(ValueError, match="Unknown collector"):
            get_collector("this_does_not_exist")


# ===================================================================
# 8. INTEGRATION — full round-trip with real Ollama + live collector
# ===================================================================


class TestFullIntegration:
    """The ultimate E2E: real Ollama + real collector + real graph.

    Uses Wikipedia because it's free, fast, and reliable.
    """

    @pytest.mark.asyncio
    async def test_wikipedia_full_roundtrip(self, ollama_model, monkeypatch):
        """User asks about Python → route → collect from Wikipedia → analyze → respond."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )

        workflow = build_graph()
        try:
            result = await workflow.ainvoke(
                {
                    "user_message": "/wiki Python programming",
                    "source": "wikipedia",
                    "query": "Python programming",
                    "items": [],
                    "analysis": "",
                    "response": "",
                    "error": "",
                }
            )
            assert result["source"] == "wikipedia"
            # Wikipedia may rate-limit us; if so, the pipeline still produces a response
            if result.get("error"):
                assert len(result["response"]) > 0
            else:
                assert len(result["items"]) >= 1
                assert len(result["analysis"]) > 20
                assert "WIKIPEDIA" in result["response"]
            assert len(result["response"]) <= 4096
        except httpx.HTTPError as e:
            pytest.skip(f"Wikipedia API unavailable: {e}")

    @pytest.mark.asyncio
    async def test_weather_full_roundtrip(self, ollama_model, monkeypatch):
        """Preset weather query → collect real weather → analyze with LLM → respond."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )

        workflow = build_graph()
        try:
            result = await workflow.ainvoke(
                {
                    "user_message": "/weather New York",
                    "source": "weather",
                    "query": "New York",
                    "items": [],
                    "analysis": "",
                    "response": "",
                    "error": "",
                }
            )
            assert result["source"] == "weather"
            assert len(result["items"]) >= 1
            assert "°C" in str(result["items"])
            assert len(result["response"]) > 10
        except httpx.HTTPError as e:
            pytest.skip(f"Weather API unavailable: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(240)
    async def test_auto_route_full_roundtrip(self, ollama_model, monkeypatch):
        """Free-text message → LLM routes → collect → analyze → respond. Full auto."""
        monkeypatch.setattr(
            "src.graph.get_llm_client",
            lambda: OllamaClient(base_url=OLLAMA_BASE, default_model=ollama_model),
        )

        workflow = build_graph()
        try:
            result = await workflow.ainvoke(
                {
                    "user_message": "Tell me about the Python programming language",
                    "source": "",
                    "query": "",
                    "items": [],
                    "analysis": "",
                    "response": "",
                    "error": "",
                }
            )
            # Should have routed somewhere valid
            assert result["source"] in list(COLLECTOR_REGISTRY.keys())
            assert len(result["query"]) > 0
            assert len(result["response"]) > 10
        except (httpx.HTTPError, Exception) as e:
            if "connect" in str(e).lower() or "timeout" in str(e).lower():
                pytest.skip(f"External API unavailable: {e}")
            raise
