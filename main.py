import asyncio
import sys

import uvicorn

from src.config import settings


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "api":
        # Run only the API server
        uvicorn.run(
            "src.api.app:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=True,
        )

    elif mode == "agent":
        # Run only the background agent
        from src.agent import run_agent

        asyncio.run(run_agent())

    elif mode == "mcp":
        # Run the MCP server
        from src.tools.mcp_server import run_mcp_server

        asyncio.run(run_mcp_server())

    elif mode == "all":
        # Run both API and agent together
        import threading

        def run_api():
            uvicorn.run(
                "src.api.app:app",
                host=settings.api_host,
                port=settings.api_port,
            )

        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()

        from src.agent import run_agent

        asyncio.run(run_agent())

    else:
        print(f"Usage: python main.py [api|agent|mcp|all]")
        print(f"  api   - Run API server only")
        print(f"  agent - Run background agent only")
        print(f"  mcp   - Run MCP server for Claude Desktop")
        print(f"  all   - Run both API and agent (default)")
        sys.exit(1)


if __name__ == "__main__":
    main()
