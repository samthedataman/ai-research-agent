"""LangGraph workflow: route → collect (with self-healing retry) → analyze → respond."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.collectors import COLLECTOR_REGISTRY, get_collector
from src.config import settings
from src.llm import get_llm_client
from src.logging_config import get_logger

logger = get_logger(__name__)

AVAILABLE_SOURCES = list(COLLECTOR_REGISTRY.keys())

# Fallback chain: if one source fails, try these in order
FALLBACK_CHAIN: dict[str, list[str]] = {
    "ddg": ["ddg_news", "news", "reddit"],
    "ddg_news": ["news", "ddg", "reddit"],
    "news": ["ddg_news", "ddg", "reddit"],
    "wikipedia": ["ddg", "news"],
    "arxiv": ["ddg", "news", "github"],
    "crypto": ["cryptonews", "ddg_news", "news"],
    "cryptonews": ["crypto", "ddg_news", "news"],
    "stocks": ["ddg", "news"],
    "reddit": ["ddg", "news"],
    "github": ["ddg", "news"],
    "weather": ["ddg"],
    "serper": ["ddg", "news"],
    "tmz": ["ddg_news", "news"],
}


class AgentState(TypedDict, total=False):
    user_message: str
    source: str
    query: str
    items: list[dict[str, Any]]
    analysis: str
    response: str
    error: str
    model: str
    analysis_model: str
    tried_sources: list[str]  # track which sources we already tried
    retry_count: int


# ── Nodes ────────────────────────────────────────────────────────────────────


async def route_node(state: AgentState) -> AgentState:
    """Determine which collector to use and what query to run.

    If source is already set (from a /command), skip LLM routing.
    Otherwise, use the LLM to pick the best source from the user's message.
    """
    if state.get("source") and state.get("query"):
        return {**state, "tried_sources": [], "retry_count": 0}

    llm = get_llm_client(state.get("model"))
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
        if "```" in text:
            text = text.split("```")[1].replace("json", "", 1).strip()
        parsed = json.loads(text)
        source = parsed.get("source", "news")
        query = parsed.get("query", state["user_message"])
        if source not in AVAILABLE_SOURCES:
            source = "news"
        return {**state, "source": source, "query": query, "tried_sources": [], "retry_count": 0}
    except Exception as e:
        logger.warning("route_fallback", error=str(e))
        return {**state, "source": "news", "query": state["user_message"], "tried_sources": [], "retry_count": 0}
    finally:
        await llm.close()


async def collect_node(state: AgentState) -> AgentState:
    """Fetch data from the selected collector. Tracks tried sources for retry."""
    source = state["source"]
    query = state["query"]
    tried = list(state.get("tried_sources", []))
    tried.append(source)

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
        if items:
            return {**state, "items": items, "error": "", "tried_sources": tried}
        else:
            logger.warning("collect_empty", source=source, query=query)
            return {**state, "items": [], "error": f"No results from {source}", "tried_sources": tried}
    except Exception as e:
        logger.error("collect_error", source=source, error=str(e))
        return {**state, "items": [], "error": f"Failed: {source} ({e})", "tried_sources": tried}


async def retry_node(state: AgentState) -> AgentState:
    """Pick the next fallback source and increment retry count."""
    tried = state.get("tried_sources", [])
    original_source = tried[0] if tried else state["source"]
    fallbacks = FALLBACK_CHAIN.get(original_source, ["news", "reddit", "ddg_news"])

    # Find first untried fallback
    next_source = None
    for fb in fallbacks:
        if fb not in tried:
            next_source = fb
            break

    if not next_source:
        # All fallbacks exhausted
        return {**state, "retry_count": 99}

    retry_count = state.get("retry_count", 0) + 1
    logger.info("retry_reroute", from_source=state["source"], to_source=next_source, attempt=retry_count)
    return {**state, "source": next_source, "error": "", "retry_count": retry_count}


async def analyze_node(state: AgentState) -> AgentState:
    """Use a large LLM to synthesize collected items into a rich Markdown briefing."""
    if state.get("error") or not state.get("items"):
        return {**state, "analysis": state.get("error", "No data to analyze.")}

    items_text = ""
    for i, item in enumerate(state["items"][:5], 1):
        items_text += f"\n--- Item {i} ---\n"
        items_text += f"Title: {item['title']}\n"
        items_text += f"Content: {item['content'][:500]}\n"
        if item.get("url"):
            items_text += f"URL: {item['url']}\n"

    # Use the analysis model (larger) — fall back to user's model or default
    analysis_model = (
        state.get("analysis_model")
        or settings.ollama_analysis_model
    )
    llm = get_llm_client(analysis_model)
    try:
        prompt = (
            f"You are a research analyst. Synthesize these {state['source']} results "
            f"for the query '{state['query']}' into a well-structured briefing.\n\n"
            f"FORMAT RULES (Telegram Markdown):\n"
            f"- Use *bold* for emphasis (NOT **bold**)\n"
            f"- Use _italic_ for secondary info\n"
            f"- Use `code` for tickers, numbers, or technical terms\n"
            f"- Use bullet points (•) for lists\n"
            f"- Include clickable links as: [Title](url)\n"
            f"- Start with a 1-2 sentence *Key Takeaway*\n"
            f"- Then list *Highlights* as bullet points\n"
            f"- End with a *Sources* section linking the URLs\n"
            f"- Keep it under 3000 characters total\n"
            f"- Do NOT use headers with # — Telegram doesn't support them\n\n"
            f"DATA:\n{items_text}"
        )
        resp = await llm.complete([{"role": "user", "content": prompt}], temperature=0.4)
        analysis = llm.get_text(resp)
        return {**state, "analysis": analysis}
    except Exception as e:
        logger.warning("analyze_fallback", error=str(e))
        fallback = f"*{state['source'].upper()} results for '{state['query']}'*\n\n"
        for item in state["items"][:5]:
            fallback += f"• *{item['title']}*\n"
            if item.get("url"):
                fallback += f"  [{item['title'][:40]}]({item['url']})\n"
        return {**state, "analysis": fallback}
    finally:
        await llm.close()


async def respond_node(state: AgentState) -> AgentState:
    """Format the analysis for Telegram (max 4096 chars)."""
    analysis = state.get("analysis", "No results available.")

    a_model = state.get("analysis_model") or settings.ollama_analysis_model
    model_tag = f" `[{a_model}]`"

    # Show if we retried
    tried = state.get("tried_sources", [])
    retry_note = ""
    if len(tried) > 1:
        retry_note = f"\n_Tried {', '.join(tried[:-1])} first, used {tried[-1]}._\n"

    header = f"🔍 *{state.get('source', 'search').upper()}* — {state.get('query', '')}{model_tag}\n{retry_note}\n"
    body = analysis

    full = header + body
    if len(full) > 4096:
        full = full[:4090] + "\n..."

    return {**state, "response": full}


# ── Graph ────────────────────────────────────────────────────────────────────


def _should_retry(state: AgentState) -> str:
    """Decide whether to retry with a fallback source or proceed to analyze."""
    has_items = bool(state.get("items"))
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    if has_items:
        return "analyze"
    if retry_count >= max_retries:
        logger.warning("retry_exhausted", retries=retry_count)
        return "analyze"  # give up, analyze with whatever we have (will show error)
    return "retry"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow with self-healing retry loop."""
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("collect", collect_node)
    graph.add_node("retry", retry_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "collect")
    # After collect: either got data → analyze, or failed → retry
    graph.add_conditional_edges("collect", _should_retry, {
        "analyze": "analyze",
        "retry": "retry",
    })
    # After retry, go back to collect with the new source
    graph.add_edge("retry", "collect")
    graph.add_edge("analyze", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
