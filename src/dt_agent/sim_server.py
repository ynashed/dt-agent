"""
sim_server.py — Isaac Sim Kit + minimal stdlib HTTP RPC server.

Runs INSIDE the Isaac Sim container. Boots Kit headless and exposes a
JSON-RPC-style HTTP endpoint on port 8765 for the agent host to invoke
tools that read or modify Kit state.

Why stdlib only
---------------
Isaac Sim's bundled Python ships dozens of pinned vendored packages
(torch._vendor.packaging, fastapi, starlette, typing_extensions, etc.)
on Kit's sys.path. Installing a modern web stack like fastmcp into that
Python invariably collides with one of those vendored copies. So this
side of the bridge speaks plain HTTP using only stdlib modules; the MCP
layer lives on the agent host, where the Python environment is clean.

Threading model
---------------
- Main thread:    SimulationApp.update() loop; drains a job queue once
                  per frame.
- HTTP threads:   ThreadingHTTPServer; each request thread posts a
                  callable to the queue and waits on its Future. No
                  omni APIs are called from request threads directly.

USD/render/Kit calls aren't thread-safe in general, so all omni work
goes through the queue and runs on the main thread.

Endpoints
---------
- GET  /tools      -> {"tools": [name, ...]}
- POST /rpc        -> body: {"tool": <name>, "args": {...}}
                      response: {"result": <json>}  or  {"error": "..."}
"""

# SimulationApp must be constructed before any other omni.* import.
from isaacsim import SimulationApp  # type: ignore

sim_app = SimulationApp({"headless": True})

import concurrent.futures  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import omni.usd  # noqa: E402  (import after SimulationApp is intentional)


# --- Thread-safe job queue (HTTP threads -> Kit main thread) ---

class JobQueue:
    """Request threads post callables; main thread drains and runs them."""

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
            except Exception as e:
                fut.set_exception(e)


jobs = JobQueue()


# --- Tool implementations (always run on Kit main thread via the queue) ---

def _impl_get_stage_info() -> dict:
    """Return the currently loaded USD stage URL and total prim count."""
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        return {"loaded": False, "url": None, "prim_count": 0}
    return {
        "loaded": True,
        "url": ctx.get_stage_url() or "",
        "prim_count": sum(1 for _ in stage.Traverse()),
    }


TOOLS = {
    "get_stage_info": _impl_get_stage_info,
}


# --- HTTP handler ---

class RPCHandler(BaseHTTPRequestHandler):
    def _json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/tools":
            self._json(200, {"tools": list(TOOLS.keys())})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/rpc":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        tool = body.get("tool")
        if tool not in TOOLS:
            self._json(400, {"error": f"unknown tool: {tool}"})
            return
        fut = jobs.submit(TOOLS[tool])
        try:
            result = fut.result(timeout=30.0)
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        self._json(200, {"result": result})

    # Silence the default per-request stderr access log; Kit's logger is
    # noisy enough without us doubling up.
    def log_message(self, format, *args):
        pass


def _run_http() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8765), RPCHandler)
    server.serve_forever()


threading.Thread(target=_run_http, daemon=True, name="rpc-http").start()


# --- Main loop ---

print(
    "[sim_server] Kit booted, HTTP RPC listening on :8765 (POST /rpc, GET /tools)",
    flush=True,
)

try:
    while sim_app.is_running():
        jobs.drain()
        sim_app.update()
finally:
    sim_app.close()
