"""
sim_client_smoke.py — Validate the FastMCP SSE pipe to the Isaac Sim container.

Runs on the host. Connects to the SSE endpoint published by sim_server.py,
lists tools, calls get_stage_info, and prints the result.

    python scripts/sim_client_smoke.py
"""
import asyncio
import sys

from fastmcp import Client

SSE_URL = "http://localhost:8765/sse"


async def main() -> int:
    try:
        async with Client(SSE_URL) as client:
            tools = await client.list_tools()
            print(f"[smoke] tools: {[t.name for t in tools]}")

            result = await client.call_tool("get_stage_info", {})
            print(f"[smoke] get_stage_info: {result.data if hasattr(result, 'data') else result}")
        return 0
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "[smoke] Is the container running? Try: docker compose up",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
