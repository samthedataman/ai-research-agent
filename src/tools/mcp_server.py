import json

from mcp.server import Server
from mcp.types import Tool, TextContent

from src.logging_config import get_logger

logger = get_logger(__name__)

server = Server("ai-research-agent")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_news",
            description="Search for recent news articles on a topic (free, no API key)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_weather",
            description="Get current weather and forecast for a location (free, no API key)",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. 'New York'"},
                },
                "required": ["location"],
            },
        ),
        Tool(
            name="get_crypto",
            description="Get crypto market data from CoinGecko (free). Query: 'trending', 'market', or a coin like 'bitcoin'",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "'trending', 'market', or coin name"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_stocks",
            description="Get stock quotes from Yahoo Finance (free). Symbols: 'AAPL,GOOGL' or 'market' for indices",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "string", "description": "Comma-separated tickers or 'market'"},
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="search_reddit",
            description="Search Reddit posts or browse subreddits (free, no API key)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term or 'r/subreddit'"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_github",
            description="Search GitHub repos or get trending repos (free, 60 req/hr)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "'trending' or search term"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_arxiv",
            description="Search for research papers on arXiv (free, no API key)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research topic or keywords"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_wikipedia",
            description="Search Wikipedia or get current events (free). Special queries: 'current_events', 'on_this_day', 'featured'",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term or special command"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="analyze_sentiment",
            description="Analyze the sentiment of a piece of text using LLM",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="search_knowledge_base",
            description="Search the agent's collected knowledge base using semantic search (RAG)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {"type": "integer", "description": "Number of results", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_digest",
            description="Get a summary digest of recent research findings",
            inputSchema={
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Hours to look back", "default": 24},
                },
            },
        ),
    ]


async def _collect_and_format(source: str, query: str, limit: int = 5) -> str:
    """Helper: collect from a source and return formatted JSON."""
    from src.collectors import get_collector

    collector = get_collector(source)
    try:
        items = await collector.collect(query, limit=limit)
        results = [
            {"title": i.title, "content": i.content, "url": i.url, "source": i.source}
            for i in items
        ]
        return json.dumps(results, indent=2)
    finally:
        await collector.close()


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    logger.info("mcp_tool_called", tool=name, arguments=arguments)

    if name == "search_news":
        text = await _collect_and_format("news", arguments["query"], arguments.get("limit", 5))
        return [TextContent(type="text", text=text)]

    elif name == "get_weather":
        text = await _collect_and_format("weather", arguments["location"])
        return [TextContent(type="text", text=text)]

    elif name == "get_crypto":
        text = await _collect_and_format("crypto", arguments["query"], arguments.get("limit", 10))
        return [TextContent(type="text", text=text)]

    elif name == "get_stocks":
        text = await _collect_and_format("stocks", arguments["symbols"])
        return [TextContent(type="text", text=text)]

    elif name == "search_reddit":
        text = await _collect_and_format("reddit", arguments["query"], arguments.get("limit", 10))
        return [TextContent(type="text", text=text)]

    elif name == "search_github":
        text = await _collect_and_format("github", arguments["query"], arguments.get("limit", 10))
        return [TextContent(type="text", text=text)]

    elif name == "search_arxiv":
        text = await _collect_and_format("arxiv", arguments["query"], arguments.get("limit", 5))
        return [TextContent(type="text", text=text)]

    elif name == "search_wikipedia":
        text = await _collect_and_format("wikipedia", arguments["query"], arguments.get("limit", 5))
        return [TextContent(type="text", text=text)]

    elif name == "analyze_sentiment":
        from src.clients.llm_factory import get_llm_client
        from src.analyzers.sentiment import SentimentAnalyzer

        llm = get_llm_client()
        try:
            analyzer = SentimentAnalyzer(llm)
            result = await analyzer.analyze(arguments["text"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        finally:
            await llm.close()

    elif name == "search_knowledge_base":
        from src.tools.vector_store import VectorStore

        store = VectorStore()
        results = store.query(
            arguments["query"], n_results=arguments.get("n_results", 5)
        )
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    elif name == "get_digest":
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8000/digest",
                json={"hours": arguments.get("hours", 24)},
            )
            return [TextContent(type="text", text=resp.text)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_mcp_server():
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
