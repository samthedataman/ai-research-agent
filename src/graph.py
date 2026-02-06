"""LangGraph workflow: route → collect → analyze → respond."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.collectors import COLLECTOR_REGISTRY, get_collector
from src.llm import get_llm_client
from src.logging_config import get_logger

logger = get_logger(__name__)

AVAILABLE_SOURCES = list(COLLECTOR_REGISTRY.keys())


class AgentState(TypedDict, total=False):
    user_message: str
    source: str
    query: str
    items: list[dict[str, Any]]
    analysis: str
    response: str
    error: str


# ── Nodes ────────────────────────────────────────────────────────────────────


async def route_node(state: AgentState) -> AgentState:
    """Determine which collector to use and what query to run.

    If source is already set (from a /command), skip LLM routing.
    Otherwise, use the LLM to pick the best source from the user's message.
    """
    if state.get("source") and state.get("query"):
        return state

    llm = get_llm_client()
    try:
        prompt = (
            f"You are a router. Given the user message, pick the best data source and extract the search query.\n"
            f"Available sources: {', '.join(AVAILABLE_SOURCES)}\n\n"
            f"User message: {state['user_message']}\n\n"
            f'Respond with ONLY valid JSON: {{"source": "...", "query": "..."}}\n'
            f"If unsure, default to source='news'."
        )
        resp = await llm.complete([{"role": "user", "content": prompt}], temperature=0.1)
        text = llm.get_text(resp).strip()
        # Parse JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1].replace("json", "", 1).strip()
        parsed = json.loads(text)
        source = parsed.get("source", "news")
        query = parsed.get("query", state["user_message"])
        if source not in AVAILABLE_SOURCES:
            source = "news"
        return {**state, "source": source, "query": query}
    except Exception as e:
        logger.warning("route_fallback", error=str(e))
        return {**state, "source": "news", "query": state["user_message"]}
    finally:
        await llm.close()


async def collect_node(state: AgentState) -> AgentState:
    """Fetch data from the selected collector."""
    source = state["source"]
    query = state["query"]

    try:
        collector = get_collector(source)
        try:
            raw_items = await collector.collect(query, limit=5)
        finally:
            await collector.close()

        items = [
            {"title": item.title, "content": item.content, "url": item.url, "source": item.source}
            for item in raw_items
        ]
        if not items:
            return {**state, "items": [], "error": f"No results found for '{query}' on {source}."}
        return {**state, "items": items}
    except Exception as e:
        logger.error("collect_error", source=source, error=str(e))
        return {**state, "items": [], "error": f"Failed to collect from {source}: {e}"}


async def analyze_node(state: AgentState) -> AgentState:
    """Use the LLM to summarize collected items into a useful briefing."""
    if state.get("error") or not state.get("items"):
        return {**state, "analysis": state.get("error", "No data to analyze.")}

    items_text = ""
    for i, item in enumerate(state["items"][:5], 1):
        items_text += f"\n{i}. **{item['title']}**\n{item['content'][:300]}\n"
        if item.get("url"):
            items_text += f"   Link: {item['url']}\n"

    llm = get_llm_client()
    try:
        prompt = (
            f"Summarize these {state['source']} results for the query '{state['query']}'. "
            f"Be concise but informative. Use bullet points. Include links where available.\n\n"
            f"{items_text}"
        )
        resp = await llm.complete([{"role": "user", "content": prompt}], temperature=0.3)
        analysis = llm.get_text(resp)
        return {**state, "analysis": analysis}
    except Exception as e:
        logger.warning("analyze_fallback", error=str(e))
        # Fallback: just format the raw items
        fallback = f"📊 *{state['source'].upper()} results for '{state['query']}'*\n\n"
        for item in state["items"][:5]:
            fallback += f"• *{item['title']}*\n"
            if item.get("url"):
                fallback += f"  {item['url']}\n"
        return {**state, "analysis": fallback}
    finally:
        await llm.close()


async def respond_node(state: AgentState) -> AgentState:
    """Format the analysis for Telegram (max 4096 chars)."""
    analysis = state.get("analysis", "No results available.")

    header = f"🔍 *{state.get('source', 'search').upper()}* — {state.get('query', '')}\n\n"
    body = analysis

    full = header + body
    # Telegram message limit
    if len(full) > 4096:
        full = full[:4090] + "\n..."

    return {**state, "response": full}


# ── Graph ────────────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("collect", collect_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "collect")
    graph.add_edge("collect", "analyze")
    graph.add_edge("analyze", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
