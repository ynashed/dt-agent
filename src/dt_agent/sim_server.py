"""
sim_server.py — Isaac Sim Kit + FastMCP-SSE server.

Runs INSIDE the Isaac Sim container. Boots Kit headless, exposes an MCP server
over SSE on port 8765, and routes tool calls onto Kit's main thread via a
thread-safe job queue.

Threading model
---------------
- Main thread:    SimulationApp.update() loop; drains job queue once per frame.
- FastMCP thread: Uvicorn/Starlette serves SSE; tools post callables to the
                  queue and await the result. No omni APIs are called from this
                  thread directly.

USD/render/Kit calls are not thread-safe in general, so all omni work goes
through the queue and runs on the main thread.
"""

# SimulationApp must be constructed before any other omni.* import.
from isaacsim import SimulationApp  # type: ignore

sim_app = SimulationApp({"headless": True})

import asyncio
import concurrent.futures
import queue
import threading

import omni.usd  # noqa: E402  (import after SimulationApp is intentional)
from fastmcp import FastMCP  # noqa: E402


# --- Thread-safe job queue ---------------------------------------------------

class JobQueue:
    """Side-thread tools post callables; main thread drains and runs them."""

    def __init__(self) -> None:
        self._q: "queue.Queue[tuple]" = queue.Queue()

    def submit(self, fn, *args, **kwargs) -> "concurrent.futures.Future":
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put((fn, args, kwargs, fut))
        return fut

    def drain(self) -> None:
        while True:
            try:
                fn, args, kwargs, fut = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as e:  # propagate to the awaiting tool
                fut.set_exception(e)


jobs = JobQueue()


# --- Main-thread tool implementations ----------------------------------------

def _impl_get_stage_info() -> dict:
    """Read current stage URL + prim count. Runs on main thread."""
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        return {"loaded": False, "url": None, "prim_count": 0}
    return {
        "loaded": True,
        "url": ctx.get_stage_url() or "",
        "prim_count": sum(1 for _ in stage.Traverse()),
    }


# --- FastMCP server (side thread) --------------------------------------------

mcp = FastMCP("isaacsim-mcp")


@mcp.tool()
async def get_stage_info() -> dict:
    """Return the currently loaded USD stage URL and total prim count."""
    return await asyncio.wrap_future(jobs.submit(_impl_get_stage_info))


def _run_mcp_server() -> None:
    # mcp.run() is blocking and starts its own asyncio loop in this thread.
    mcp.run(transport="sse", host="0.0.0.0", port=8765)


threading.Thread(target=_run_mcp_server, daemon=True, name="fastmcp-sse").start()


# --- Main loop ---------------------------------------------------------------

print("[sim_server] Kit booted, FastMCP-SSE listening on :8765/sse", flush=True)

try:
    while sim_app.is_running():
        jobs.drain()
        sim_app.update()
finally:
    sim_app.close()
