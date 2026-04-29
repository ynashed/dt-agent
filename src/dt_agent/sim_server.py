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

# Kit's extensions prepend their bundled pip_prebundle dirs to sys.path during
# boot, which shadows the (newer) transitive deps we installed to /opt/mcp-site
# — most notably typing_extensions, where Kit ships a version older than the
# one fastmcp's chain (key_value -> pydantic adapter) needs (TypeForm wasn't
# added until typing_extensions 4.13). Re-prepend MCP_SITE and evict any stale
# modules so the fastmcp import resolves against the right versions.
import os  # noqa: E402
import sys  # noqa: E402

_mcp_site = os.environ.get("MCP_SITE", "/opt/mcp-site")
if _mcp_site in sys.path:
    sys.path.remove(_mcp_site)
sys.path.insert(0, _mcp_site)
for _stale in ("typing_extensions",):
    sys.modules.pop(_stale, None)

import asyncio  # noqa: E402
import concurrent.futures  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402

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
