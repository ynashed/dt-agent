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
import os  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import omni.usd  # noqa: E402  (import after SimulationApp is intentional)
from pxr import Gf, Usd, UsdGeom  # noqa: E402


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

def _stage():
    return omni.usd.get_context().get_stage()


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


def _impl_query_stage(prim_path: str = "/", depth: int = 3) -> dict:
    """List child prims under prim_path, limited to `depth` levels.

    Each prim entry includes its path, type, and (if Xformable) its
    world-space translation. Returns at most a few thousand prims to
    keep responses manageable.
    """
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return {"error": f"prim not found: {prim_path}"}

    prims: list[dict] = []
    MAX = 5000

    def walk(prim, current_depth: int) -> None:
        if len(prims) >= MAX:
            return
        info = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName() or "",
        }
        xf = UsdGeom.Xformable(prim)
        if xf:
            try:
                m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = m.ExtractTranslation()
                info["translate"] = [float(t[0]), float(t[1]), float(t[2])]
            except Exception:
                pass
        prims.append(info)
        if current_depth >= depth:
            return
        for child in prim.GetChildren():
            walk(child, current_depth + 1)

    walk(root, 0)
    return {
        "root": str(root.GetPath()),
        "prims": prims,
        "truncated": len(prims) >= MAX,
    }


def _impl_create_primitive(prim_path: str, prim_type: str = "Xform") -> dict:
    """Create a USD prim of `prim_type` (Xform / Cube / Sphere / Cylinder /
    Cone / Capsule / Plane) at `prim_path`. Existing prims are left alone."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    prim = stage.DefinePrim(prim_path, prim_type)
    if not prim.IsValid():
        return {"error": f"failed to create {prim_type} at {prim_path}"}
    return {"ok": True, "prim_path": str(prim.GetPath()), "type": prim_type}


def _impl_add_reference_to_stage(usd_path: str, prim_path: str) -> dict:
    """Add `usd_path` as a reference under `prim_path`. Creates an Xform
    at `prim_path` first if it doesn't exist."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
        if not prim.IsValid():
            return {"error": f"failed to create container prim at {prim_path}"}
    ok = prim.GetReferences().AddReference(usd_path)
    return {"ok": bool(ok), "prim_path": prim_path, "usd_path": usd_path}


def _impl_set_transform(
    prim_path: str,
    translate: list | None = None,
    rotate: list | None = None,
    scale: list | None = None,
) -> dict:
    """Set translate (xyz, meters), rotate (xyz Euler degrees), and/or
    scale on a prim. Any of the three may be omitted."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return {"error": f"prim not found: {prim_path}"}
    if not UsdGeom.Xformable(prim):
        return {"error": f"prim is not Xformable: {prim_path}"}

    api = UsdGeom.XformCommonAPI(prim)
    if translate is not None:
        api.SetTranslate(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if rotate is not None:
        api.SetRotate(
            Gf.Vec3f(float(rotate[0]), float(rotate[1]), float(rotate[2])),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
    if scale is not None:
        api.SetScale(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
    return {"ok": True, "prim_path": prim_path}


def _impl_save_stage(file_path: str) -> dict:
    """Save the current stage to `file_path` inside the container.
    `file_path` must be on a writable mount; `/workspace/dt-agent/output/...`
    is the convention (mount that on the host to inspect saved USDs)."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ok = stage.Export(file_path)
    return {"ok": bool(ok), "file_path": file_path}


# Default search roots for shipped content inside the Isaac Sim 5.1 image.
# Most "official" robot/prop USDs stream from Nucleus or the OpenUSD CDN, but
# extensions and their `data/` dirs hold useful demo assets that work offline.
_DEFAULT_ASSET_ROOTS = [
    "/isaac-sim/exts",
    "/isaac-sim/extscache",
    "/workspace/dt-agent/assets",
]
_USD_EXTS = (".usd", ".usda", ".usdc", ".usdz")

# Curated catalog of HTTPS URLs on NVIDIA's OpenUSD CDN. Loaded once at
# startup; entries get returned alongside filesystem matches by search_assets.
_CATALOG_PATH = "/workspace/dt-agent/catalog/asset_catalog.json"


def _load_catalog() -> dict:
    try:
        with open(_CATALOG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"assets": []}
    except Exception as e:
        print(
            f"[sim_server] WARN: failed to load catalog at {_CATALOG_PATH}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return {"assets": []}


_CATALOG = _load_catalog()


def _catalog_haystack(entry: dict) -> str:
    return " ".join(
        [
            entry.get("name", ""),
            entry.get("description", ""),
            entry.get("category", ""),
            " ".join(entry.get("tags", [])),
            entry.get("url", ""),
        ]
    ).lower()


def _impl_search_assets(
    query: str,
    limit: int = 30,
    roots: list | None = None,
    sources: list | None = None,
) -> dict:
    """Case-insensitive substring search for USDs across the curated catalog
    and the local filesystem. Returns merged matches as objects.

    Args:
        query: substring to match (case-insensitive). Empty string returns
               everything up to `limit`.
        limit: total cap on matches across both sources.
        roots: filesystem roots to walk. Defaults to a curated set of in-image
               dirs plus /workspace/dt-agent/assets.
        sources: subset of {"catalog", "filesystem"} to search; both by default.

    Returns: {"matches": [{path, source, ...}], "truncated": bool}
        - source="catalog" entries also include name, description, category,
          and verified flag from the catalog file.
        - source="filesystem" entries are minimal: just {path, source}.
    """
    sources = sources or ["catalog", "filesystem"]
    q = (query or "").lower()
    matches: list[dict] = []

    if "catalog" in sources:
        for entry in _CATALOG.get("assets", []):
            if q and q not in _catalog_haystack(entry):
                continue
            matches.append(
                {
                    "path": entry.get("url", ""),
                    "source": "catalog",
                    "name": entry.get("name"),
                    "description": entry.get("description"),
                    "category": entry.get("category"),
                    "verified": entry.get("verified", False),
                }
            )
            if len(matches) >= limit:
                return {"matches": matches, "truncated": True}

    if "filesystem" in sources:
        roots = roots or _DEFAULT_ASSET_ROOTS
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if not fn.lower().endswith(_USD_EXTS):
                        continue
                    full = os.path.join(dirpath, fn)
                    if q and q not in full.lower():
                        continue
                    matches.append({"path": full, "source": "filesystem"})
                    if len(matches) >= limit:
                        return {"matches": matches, "truncated": True}

    return {"matches": matches, "truncated": False}


TOOLS = {
    "get_stage_info": _impl_get_stage_info,
    "query_stage": _impl_query_stage,
    "create_primitive": _impl_create_primitive,
    "add_reference_to_stage": _impl_add_reference_to_stage,
    "set_transform": _impl_set_transform,
    "save_stage": _impl_save_stage,
    "search_assets": _impl_search_assets,
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
        args = body.get("args", {}) or {}
        if not isinstance(args, dict):
            self._json(400, {"error": "args must be an object"})
            return
        fut = jobs.submit(TOOLS[tool], **args)
        try:
            result = fut.result(timeout=60.0)
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
print(f"[sim_server] tools: {sorted(TOOLS.keys())}", flush=True)
print(
    f"[sim_server] catalog: {len(_CATALOG.get('assets', []))} entries "
    f"loaded from {_CATALOG_PATH}",
    flush=True,
)

try:
    while sim_app.is_running():
        jobs.drain()
        sim_app.update()
finally:
    sim_app.close()
