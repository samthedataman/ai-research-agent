import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.graph import AgentState, route_node, collect_node, analyze_node, respond_node


class TestRouteNode:
    @pytest.mark.asyncio
    async def test_passthrough_when_source_set(self):
        """If source and query are already set (from /command), skip LLM routing."""
        state: AgentState = {
            "user_message": "/news AI agents",
            "source": "news",
            "query": "AI agents",
            "items": [],
            "analysis": "",
            "response": "",
            "error": "",
        }
        result = await route_node(state)
        assert result["source"] == "news"
        assert result["query"] == "AI agents"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """If the LLM fails, default to news source."""
        state: AgentState = {
            "user_message": "What's happening in tech?",
            "source": "",
            "query": "",
            "items": [],
            "analysis": "",
            "response": "",
            "error": "",
        }
        with patch("src.graph.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.complete.side_effect = Exception("LLM down")
            mock_client.close = AsyncMock()
            mock_llm.return_value = mock_client

            result = await route_node(state)
            assert result["source"] == "news"
            assert result["query"] == "What's happening in tech?"


class TestCollectNode:
    @pytest.mark.asyncio
    async def test_collect_success(self):
        """Items are collected and converted to dicts."""
        state: AgentState = {
            "user_message": "/news AI",
            "source": "news",
            "query": "AI",
            "items": [],
            "analysis": "",
            "response": "",
            "error": "",
        }
        mock_item = MagicMock()
        mock_item.title = "Test Article"
        mock_item.content = "Article content"
        mock_item.url = "https://example.com"
        mock_item.source = "news"

        with patch("src.graph.get_collector") as mock_get:
            mock_collector = AsyncMock()
            mock_collector.collect = AsyncMock(return_value=[mock_item])
            mock_collector.close = AsyncMock()
            mock_get.return_value = mock_collector

            result = await collect_node(state)
            assert len(result["items"]) == 1
            assert result["items"][0]["title"] == "Test Article"

    @pytest.mark.asyncio
    async def test_collect_empty(self):
        """No results sets an error message."""
        state: AgentState = {
            "user_message": "/news nothing",
            "source": "news",
            "query": "nothing",
            "items": [],
            "analysis": "",
            "response": "",
            "error": "",
        }
        with patch("src.graph.get_collector") as mock_get:
            mock_collector = AsyncMock()
            mock_collector.collect = AsyncMock(return_value=[])
            mock_collector.close = AsyncMock()
            mock_get.return_value = mock_collector

            result = await collect_node(state)
            assert result["items"] == []
            assert "No results" in result["error"]


class TestRespondNode:
    @pytest.mark.asyncio
    async def test_format_response(self):
        """Response includes header and analysis."""
        state: AgentState = {
            "user_message": "/news AI",
            "source": "news",
            "query": "AI",
            "items": [],
            "analysis": "Here are the top AI stories.",
            "response": "",
            "error": "",
        }
        result = await respond_node(state)
        assert "NEWS" in result["response"]
        assert "Here are the top AI stories." in result["response"]

    @pytest.mark.asyncio
    async def test_truncate_long_response(self):
        """Response over 4096 chars gets truncated."""
        state: AgentState = {
            "user_message": "/news AI",
            "source": "news",
            "query": "AI",
            "items": [],
            "analysis": "x" * 5000,
            "response": "",
            "error": "",
        }
        result = await respond_node(state)
        assert len(result["response"]) <= 4096
