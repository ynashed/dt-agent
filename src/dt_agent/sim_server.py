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

import os

# SimulationApp must be constructed before any other omni.* import.
from isaacsim import SimulationApp  # type: ignore

# Optional WebRTC livestream — boots Kit with the streaming experience so a
# remote client (Isaac Sim WebRTC Streaming Client at 127.0.0.1) can watch
# the scene live as the agent edits. Requires the container to run with
# network_mode: host (WebRTC negotiation does not work through bridged
# networking). Toggle via DT_AGENT_LIVESTREAM=1 in the compose env.
_LIVESTREAM = os.environ.get("DT_AGENT_LIVESTREAM", "0") == "1"
_STREAMING_EXPERIENCE = "/isaac-sim/apps/isaacsim.exp.full.streaming.kit"

if _LIVESTREAM:
    print(f"[sim_server] livestream enabled — loading {_STREAMING_EXPERIENCE}")
    print("[sim_server] connect via Isaac Sim WebRTC Streaming Client at 127.0.0.1")
    sim_app = SimulationApp({"headless": True}, experience=_STREAMING_EXPERIENCE)
else:
    sim_app = SimulationApp({"headless": True})

import concurrent.futures  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import omni.usd  # noqa: E402  (import after SimulationApp is intentional)
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402


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


def _ensure_xform_ancestors(stage, prim_path: str) -> None:
    """Make sure every ancestor of `prim_path` exists as an Xform prim.

    USD's `DefinePrim` will auto-create missing intermediates as typeless
    prims, which are NOT Xformable — so `set_transform` on the parent of
    a referenced asset fails with "prim is not Xformable". Walk the chain
    and define each missing or typeless ancestor as Xform.
    """
    parts = [p for p in prim_path.split("/") if p]
    cur = ""
    for part in parts[:-1]:
        cur = cur + "/" + part
        existing = stage.GetPrimAtPath(cur)
        if not existing.IsValid():
            stage.DefinePrim(cur, "Xform")
        elif not existing.GetTypeName():
            existing.SetTypeName("Xform")


def _impl_create_primitive(prim_path: str, prim_type: str = "Xform") -> dict:
    """Create a USD prim of `prim_type` (Xform / Cube / Sphere / Cylinder /
    Cone / Capsule / Plane) at `prim_path`. Existing prims are left alone."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    _ensure_xform_ancestors(stage, prim_path)
    prim = stage.DefinePrim(prim_path, prim_type)
    if not prim.IsValid():
        return {"error": f"failed to create {prim_type} at {prim_path}"}
    return {"ok": True, "prim_path": str(prim.GetPath()), "type": prim_type}


def _impl_add_reference_to_stage(usd_path: str, prim_path: str) -> dict:
    """Add `usd_path` as a reference under `prim_path`. Creates an Xform
    at `prim_path` first if it doesn't exist; ancestors are also Xform."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    _ensure_xform_ancestors(stage, prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
        if not prim.IsValid():
            return {"error": f"failed to create container prim at {prim_path}"}
    ok = prim.GetReferences().AddReference(usd_path)
    # Many Isaac assets (Franka, UR10, others) compose their meshes via
    # `payload:` arcs. AddReference alone doesn't auto-load payloads on a
    # running stage — without Load(), the prim stays geometryless forever
    # and get_prim_bounds returns empty boxes. LoadWithDescendants is the
    # default, so this pulls in the whole asset subtree.
    stage.Load(prim.GetPath())
    return {"ok": bool(ok), "prim_path": prim_path, "usd_path": usd_path}


def _impl_delete_prim(prim_path: str) -> dict:
    """Remove a prim and all its descendants from the stage. Use to clean
    up probe assets after measurement, or to undo a misplaced reference."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    if prim_path in ("", "/", "/World"):
        return {"error": f"refusing to delete protected path: {prim_path!r}"}
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return {"error": f"prim not found: {prim_path}"}
    ok = stage.RemovePrim(prim_path)
    return {"ok": bool(ok), "prim_path": prim_path}


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

    # XformCommonAPI is the fast path but throws on prims whose references
    # carry an incompatible xform stack (e.g. a matrix op). Fall back to
    # direct Xformable ops in that case — they write to the current layer
    # and override the reference's stack.
    try:
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
    except Exception:
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        if translate is not None:
            xf.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
        if rotate is not None:
            xf.AddRotateXYZOp().Set(Gf.Vec3f(float(rotate[0]), float(rotate[1]), float(rotate[2])))
        if scale is not None:
            xf.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
    return {"ok": True, "prim_path": prim_path}


def _impl_save_stage(file_path: str) -> dict:
    """Save the current stage's root layer to `file_path`.

    Saves only the root layer (references stay as references) rather than
    flattening the whole stage. stage.Export() inlines every referenced mesh,
    turning a simple scene into multi-GB files. GetRootLayer().Export() keeps
    the file small and the references resolvable from their CDN URLs.
    """
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ok = stage.GetRootLayer().Export(file_path)
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


# --- Viewport capture ---------------------------------------------------
#
# Phase 1.5 — render a fixed-pose observation camera to a PNG file the agent
# host (and the VLM observer) can consume. We use omni.replicator.core's
# render_product + LdrColor annotator to drive the render in headless mode,
# and Pillow (bundled with Isaac Sim) to encode the PNG. Render product +
# annotator are cached at module level so repeat captures don't leak Kit
# resources.

DEFAULT_OBSERVATION_CAMERA = "/World/_dt_observation_cam"
DEFAULT_OBSERVATION_LIGHT = "/World/_dt_observation_light"
# Default 3/4 view of the workcell-shaped neighborhood near origin. Eye and
# target are overridable per call via kwargs, but for the fixed-pose mode
# the agent is expected to hit these defaults so the same view repeats.
DEFAULT_CAMERA_EYE = (3.0, 3.0, 2.0)
DEFAULT_CAMERA_TARGET = (0.6, 0.0, 0.45)
# 18mm gives ~60° horizontal FOV on the default 21mm aperture — wide enough
# to comfortably frame the workcell from (3,3,2). 24mm (~47°) cropped the
# smoke sphere at (1,2,0.5).
DEFAULT_FOCAL_LENGTH_MM = 18.0
DEFAULT_DOME_INTENSITY = 300.0
DEFAULT_RESOLUTION = (1280, 720)
CAPTURE_OUTPUT_DIR = "/workspace/dt-agent/output/captures"

_capture_state = {
    "render_product": None,
    "annotator": None,
    "camera_path": None,
    "resolution": None,
}


def _look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)) -> "Gf.Matrix4d":
    """Build a camera-to-world Matrix4d so a USD camera at `eye` looks at
    `target` with the given world up. USD cameras face their local -Z."""
    eye_v = Gf.Vec3d(*eye)
    target_v = Gf.Vec3d(*target)
    up_v = Gf.Vec3d(*up)
    forward = (target_v - eye_v).GetNormalized()
    right = Gf.Cross(forward, up_v).GetNormalized()
    true_up = Gf.Cross(right, forward).GetNormalized()
    return Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        true_up[0], true_up[1], true_up[2], 0.0,
        -forward[0], -forward[1], -forward[2], 0.0,
        eye_v[0], eye_v[1], eye_v[2], 1.0,
    )


def _ensure_observation_camera(camera_path: str, eye, target) -> None:
    """Create the camera prim if missing, then (re)set its transform AND
    intrinsics on every call. Refreshing on every capture means tweaks to
    DEFAULT_FOCAL_LENGTH_MM take effect after a `docker compose restart`
    with no manual prim cleanup."""
    stage = _stage()
    if stage is None:
        raise RuntimeError("no stage loaded")
    if not stage.GetPrimAtPath(camera_path).IsValid():
        UsdGeom.Camera.Define(stage, camera_path)
    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    cam.GetFocalLengthAttr().Set(DEFAULT_FOCAL_LENGTH_MM)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
    xform = UsdGeom.Xformable(cam)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(_look_at_matrix(eye, target))


def _ensure_default_lighting(light_path: str = DEFAULT_OBSERVATION_LIGHT) -> None:
    """Create a DomeLight at `light_path` if missing, then (re)set its
    intensity and color on every call so tweaking DEFAULT_DOME_INTENSITY
    takes effect after a `docker compose restart` with no manual prim
    cleanup."""
    stage = _stage()
    if stage is None:
        raise RuntimeError("no stage loaded")
    if not stage.GetPrimAtPath(light_path).IsValid():
        UsdLux.DomeLight.Define(stage, light_path)
    dome = UsdLux.DomeLight(stage.GetPrimAtPath(light_path))
    intensity_attr = dome.GetIntensityAttr() or dome.CreateIntensityAttr()
    intensity_attr.Set(DEFAULT_DOME_INTENSITY)
    color_attr = dome.GetColorAttr() or dome.CreateColorAttr()
    color_attr.Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _impl_capture_viewport(
    camera_path: str | None = None,
    eye: list | None = None,
    target: list | None = None,
    resolution: list | None = None,
    file_path: str | None = None,
) -> dict:
    """Render the scene from an observation camera and save a PNG.

    With all args defaulted, captures from the fixed-pose default camera
    (`/World/_dt_observation_cam`) — created on first call, transform
    refreshed on each call. Pass `camera_path` to use an already-existing
    camera prim instead. `eye` / `target` only affect the default camera;
    explicit `camera_path` uses whatever transform that prim already has.
    `file_path` defaults to a timestamped name under
    `/workspace/dt-agent/output/captures/`.
    """
    import omni.replicator.core as rep  # imported lazily; avoids touching
                                        # replicator at module load if no
                                        # one is using this tool.

    using_default = camera_path is None
    cam_path = camera_path or DEFAULT_OBSERVATION_CAMERA
    eye_v = tuple(eye) if eye is not None else DEFAULT_CAMERA_EYE
    target_v = tuple(target) if target is not None else DEFAULT_CAMERA_TARGET
    res = tuple(resolution) if resolution is not None else DEFAULT_RESOLUTION

    if using_default:
        _ensure_observation_camera(cam_path, eye_v, target_v)
    else:
        # Verify the requested camera exists.
        prim = _stage().GetPrimAtPath(cam_path)
        if not prim.IsValid():
            return {"error": f"camera prim not found: {cam_path}"}

    # Make sure something illuminates the scene. The default Isaac Sim stage
    # ships without a light; without one, captures of our Cube-primitive
    # workcell are uniformly near-black.
    _ensure_default_lighting()

    # (Re)build the render product if camera or resolution changed.
    if (
        _capture_state["camera_path"] != cam_path
        or _capture_state["resolution"] != res
        or _capture_state["render_product"] is None
    ):
        _capture_state["render_product"] = rep.create.render_product(cam_path, res)
        _capture_state["annotator"] = rep.AnnotatorRegistry.get_annotator("LdrColor")
        _capture_state["annotator"].attach([_capture_state["render_product"]])
        _capture_state["camera_path"] = cam_path
        _capture_state["resolution"] = res

    # rep.orchestrator.step() returns before the render fully lands, so we
    # bracket it with extra Kit updates to let textures, materials, lights,
    # and HTTPS-streamed asset references finish committing to the buffer.
    for _ in range(5):
        sim_app.update()
    rep.orchestrator.step()
    for _ in range(15):
        sim_app.update()

    data = _capture_state["annotator"].get_data()
    if data is None or getattr(data, "size", 0) == 0:
        return {"error": "annotator returned empty data; render may not have completed"}

    # Diagnostic — without this, "blank" results are indistinguishable from
    # "no lighting", "empty stage", "wrong AOV", or "render not ready".
    try:
        print(
            f"[sim_server] capture stats: shape={tuple(data.shape)} "
            f"dtype={data.dtype} min={int(data.min())} max={int(data.max())} "
            f"mean={float(data.mean()):.2f}",
            flush=True,
        )
    except Exception:
        pass

    if file_path is None:
        os.makedirs(CAPTURE_OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(
            CAPTURE_OUTPUT_DIR, f"capture_{int(time.time() * 1000)}.png"
        )
    else:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    try:
        from PIL import Image
        mode = "RGBA" if data.shape[-1] == 4 else "RGB"
        Image.fromarray(data, mode=mode).save(file_path)
    except ImportError:
        return {"error": "PIL/Pillow not available in Kit's bundled Python"}
    except Exception as e:
        return {"error": f"PNG encode failed: {type(e).__name__}: {e}"}

    return {
        "ok": True,
        "file_path": file_path,
        "camera_path": cam_path,
        "resolution": [int(res[0]), int(res[1])],
        "size": [int(data.shape[1]), int(data.shape[0])],
        "stats": {
            "min": int(data.min()),
            "max": int(data.max()),
            "mean": round(float(data.mean()), 2),
        },
    }


def _impl_get_prim_bounds(prim_path: str) -> dict:
    """Return the world-space axis-aligned bounding box (min, max, size) of a
    prim. Use after add_reference_to_stage to measure a loaded asset's
    dimensions before tiling it."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return {"error": f"prim not found: {prim_path}"}
    # Pump a few frames so recently-referenced assets have had time to resolve.
    for _ in range(5):
        sim_app.update()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    bound = cache.ComputeWorldBound(prim)
    rng = bound.GetRange()
    if rng.IsEmpty():
        # Differentiate "asset failed to compose" from "still loading" so the
        # agent doesn't waste iterations re-trying a prim that will never
        # resolve. Immediate children are enough signal — a working reference
        # always brings in at least one child.
        if not any(prim.GetChildren()):
            return {"error": (
                f"prim has no descendants at {prim_path} — the reference "
                "appears to have failed to compose. Verify the usd_path is "
                "reachable and points at a USD with a valid default prim. "
                "Retrying will not help."
            )}
        return {"error": (
            f"prim at {prim_path} has descendants but no renderable geometry "
            "in [default, render] purposes — children may be Xform-only, "
            "guide-purpose, or proxy-purpose."
        )}
    mn, mx = rng.GetMin(), rng.GetMax()
    sz = mx - mn
    return {
        "prim_path": prim_path,
        "min": [round(float(mn[i]), 4) for i in range(3)],
        "max": [round(float(mx[i]), 4) for i in range(3)],
        "size": [round(float(sz[i]), 4) for i in range(3)],
    }


_LIGHT_TYPE_MAP = {
    "SphereLight": UsdLux.SphereLight,
    "RectLight": UsdLux.RectLight,
    "DiskLight": UsdLux.DiskLight,
    "CylinderLight": UsdLux.CylinderLight,
}


def _impl_add_light(
    prim_path: str,
    light_type: str = "SphereLight",
    intensity: float = 3000.0,
    translate: list | None = None,
    color: list | None = None,
    radius: float | None = None,
    width: float | None = None,
    height: float | None = None,
) -> dict:
    """Create a USD light prim. Use for interior lighting in enclosed spaces
    (rooms, warehouses) where the DomeLight from capture_viewport cannot
    illuminate through walls/ceiling.

    light_type: SphereLight | RectLight | DiskLight | CylinderLight
    intensity:  photometric intensity. 3000–10000 for warehouse-scale spaces.
    radius:     SphereLight/DiskLight radius in metres.
    width/height: RectLight dimensions in metres.
    """
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}
    if light_type not in _LIGHT_TYPE_MAP:
        return {"error": f"unknown light_type '{light_type}'. Valid: {list(_LIGHT_TYPE_MAP)}"}
    _ensure_xform_ancestors(stage, prim_path)
    light = _LIGHT_TYPE_MAP[light_type].Define(stage, prim_path)
    if not light:
        return {"error": f"failed to create {light_type} at {prim_path}"}
    light.CreateIntensityAttr().Set(float(intensity))
    if color is not None:
        light.CreateColorAttr().Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
    if radius is not None and hasattr(light, "CreateRadiusAttr"):
        light.CreateRadiusAttr().Set(float(radius))
    if width is not None and hasattr(light, "CreateWidthAttr"):
        light.CreateWidthAttr().Set(float(width))
    if height is not None and hasattr(light, "CreateHeightAttr"):
        light.CreateHeightAttr().Set(float(height))
    if translate is not None:
        UsdGeom.XformCommonAPI(light).SetTranslate(
            Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2]))
        )
    return {"ok": True, "prim_path": prim_path, "light_type": light_type, "intensity": intensity}


def _impl_bind_material(prim_path: str, material_url: str) -> dict:
    """Create an MDL material prim under /World/Looks/ (reusing it if it
    already exists) and bind it to the prim at prim_path."""
    stage = _stage()
    if stage is None:
        return {"error": "no stage loaded"}

    target = stage.GetPrimAtPath(prim_path)
    if not target.IsValid():
        return {"error": f"prim not found: {prim_path}"}

    # Derive a stable prim name from the MDL filename.
    mdl_name = material_url.rsplit("/", 1)[-1]          # e.g. Steel_Stainless.mdl
    prim_name = mdl_name.replace(".", "_").replace("-", "_")  # Steel_Stainless_mdl
    material_prim_path = f"/World/Looks/{prim_name}"

    mat_prim = stage.GetPrimAtPath(material_prim_path)
    if not mat_prim.IsValid():
        import omni.kit.commands  # noqa: PLC0415
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=material_url,
            mtl_name=mdl_name.rsplit(".", 1)[0],   # module name without extension
            mtl_path=material_prim_path,
            select_new_prim=False,
        )
        mat_prim = stage.GetPrimAtPath(material_prim_path)
        if not mat_prim.IsValid():
            return {"error": f"CreateMdlMaterialPrim failed for {material_url}"}

    from pxr import UsdShade  # noqa: PLC0415
    material = UsdShade.Material(mat_prim)
    UsdShade.MaterialBindingAPI.Apply(target).Bind(
        material, UsdShade.Tokens.strongerThanDescendants
    )
    return {"ok": True, "prim_path": prim_path, "material_prim_path": material_prim_path}


def _impl_load_stage(file_path: str) -> dict:
    """Open `file_path` as the working stage, replacing whatever is loaded.
    Used by chat_tasks.py at startup to bring an authored scene into Kit
    before the task agent starts scripting against it."""
    if not os.path.isfile(file_path):
        return {"ok": False, "error": f"file not found: {file_path}"}
    ctx = omni.usd.get_context()
    result = ctx.open_stage(file_path)
    # open_stage may return either a bool or (success, msg) depending on Kit version.
    if isinstance(result, tuple):
        success, msg = result[0], (result[1] if len(result) > 1 else "")
    else:
        success, msg = bool(result), ""
    if not success:
        return {"ok": False, "error": f"open_stage failed: {msg}"}
    # Pump a few frames so the load completes before we report.
    for _ in range(5):
        sim_app.update()
    stage = ctx.get_stage()
    prim_count = sum(1 for _ in stage.Traverse()) if stage else 0
    return {"ok": True, "file_path": file_path, "prim_count": prim_count}


VIDEO_OUTPUT_DIR = "/workspace/dt-agent/output/captures"
# Lower fps + lower resolution than capture_viewport stills, because the
# Qwen3VL processor in the Cosmos Reason NIM 400s on the larger payloads
# we were producing at 10 fps × 1280×720. 3 fps × 640×360 keeps total
# video tokens well below the processor's implicit cap while still giving
# the VLM enough frames to perceive motion across a typical task.
VIDEO_TARGET_FPS = 3
VIDEO_RESOLUTION = (640, 360)

# Separate render product from capture_viewport's higher-res still product
# (DEFAULT_RESOLUTION). Keeping them separate avoids recreating products
# on every alternation between observe (still) and run_python (video).
_video_capture_state = {
    "render_product": None,
    "annotator": None,
    "camera_path": None,
}


def _impl_run_python(script_path: str) -> dict:
    """Execute a Python script on Kit's main thread, recording an mp4 of the
    observation camera throughout the script's execution.

    The script runs with full access to Kit's bundled Python — it can
    import omni, pxr, manipulate the stage, and call sim_app.update() /
    world.step() to advance the renderer/physics. stdout/stderr are
    captured. Exceptions are caught and the traceback returned in `error`
    so the agent can self-correct without crashing the server.

    Video recording is automatic: sim_app.update is patched to grab a
    rendered frame from the default observation camera at ~VIDEO_TARGET_FPS,
    so the agent sees the actual trajectory rather than just the final
    pose. The mp4 path is returned in `video_path`.

    PoC scope: no sandboxing. Single-user only.

    Returns: {ok, stdout, stderr, error?, elapsed_s, script_path,
              video_path?, video_frame_count}
    """
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415
    import traceback  # noqa: PLC0415

    if not os.path.isfile(script_path):
        return {"ok": False, "error": f"script not found: {script_path}"}

    try:
        with open(script_path) as f:
            source = f.read()
        code = compile(source, script_path, "exec")
    except SyntaxError as e:
        return {"ok": False, "error": f"SyntaxError: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"failed to load script: {type(e).__name__}: {e}"}

    # ── Video recording setup ────────────────────────────────────────────
    try:
        import cv2  # noqa: PLC0415
        import omni.replicator.core as rep  # noqa: PLC0415
    except Exception as e:
        return {"ok": False, "error": f"video deps missing: {type(e).__name__}: {e}"}

    cam_path = DEFAULT_OBSERVATION_CAMERA
    _ensure_observation_camera(cam_path, DEFAULT_CAMERA_EYE, DEFAULT_CAMERA_TARGET)
    _ensure_default_lighting()

    # Video gets its own render product at VIDEO_RESOLUTION, separate from
    # capture_viewport's still product at DEFAULT_RESOLUTION.
    if (
        _video_capture_state["render_product"] is None
        or _video_capture_state["camera_path"] != cam_path
    ):
        if _video_capture_state["render_product"] is not None:
            try:
                _video_capture_state["annotator"].detach(
                    [_video_capture_state["render_product"]]
                )
            except Exception:
                pass
        _video_capture_state["render_product"] = rep.create.render_product(
            cam_path, VIDEO_RESOLUTION
        )
        _video_capture_state["annotator"] = rep.AnnotatorRegistry.get_annotator("LdrColor")
        _video_capture_state["annotator"].attach(
            [_video_capture_state["render_product"]]
        )
        _video_capture_state["camera_path"] = cam_path
    annotator = _video_capture_state["annotator"]

    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(VIDEO_OUTPUT_DIR, f"run_{int(time.time() * 1000)}.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(VIDEO_TARGET_FPS),
        (int(VIDEO_RESOLUTION[0]), int(VIDEO_RESOLUTION[1])),
    )
    if not writer.isOpened():
        return {"ok": False, "error": f"failed to open VideoWriter at {video_path}"}

    # Warm-up renders so the annotator buffer is populated before the
    # subscription starts (so the first grab gets a real frame).
    for _ in range(5):
        sim_app.update()
    rep.orchestrator.step()
    for _ in range(5):
        sim_app.update()

    grab_state = {
        "last": 0.0,
        "interval": 1.0 / VIDEO_TARGET_FPS,
        "frames": 0,
        "in_capture": False,  # re-entry guard: rep.orchestrator.step pumps Kit updates
    }

    def _grab_frame():
        try:
            rep.orchestrator.step()
            data = annotator.get_data()
            if data is None or getattr(data, "size", 0) == 0:
                return
            bgr = (
                cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
                if data.shape[-1] == 4
                else cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            )
            writer.write(bgr)
            grab_state["frames"] += 1
        except Exception as e:
            print(f"[sim_server] WARN: video grab failed: {type(e).__name__}: {e}", flush=True)

    # Subscribe to Kit's update event stream instead of monkey-patching
    # sim_app.update — scripts using omni.isaac.core.World.step() never
    # call sim_app.update directly, so the previous patch saw only the
    # final post-exec grab. Kit's update event fires on every tick
    # regardless of who triggered it.
    import omni.kit.app  # noqa: PLC0415
    update_stream = omni.kit.app.get_app().get_update_event_stream()

    def _on_kit_update(_event):
        if grab_state["in_capture"]:
            return
        now = time.monotonic()
        if now - grab_state["last"] >= grab_state["interval"]:
            grab_state["in_capture"] = True
            grab_state["last"] = now
            try:
                _grab_frame()
            finally:
                grab_state["in_capture"] = False

    sub = update_stream.create_subscription_to_pop(
        _on_kit_update, name="dt_agent_video_grab"
    )

    # ── Exec ─────────────────────────────────────────────────────────────
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    script_globals: dict = {
        "__name__": "__script__",
        "__file__": script_path,
    }

    start = time.time()
    error: str | None = None
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, script_globals)  # noqa: S102
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
    elapsed = time.time() - start

    # ── Cleanup ──────────────────────────────────────────────────────────
    # Dropping the subscription carrier releases it.
    sub = None  # noqa: F841
    # One final grab so the post-script state is in the video.
    grab_state["in_capture"] = True
    try:
        _grab_frame()
    finally:
        grab_state["in_capture"] = False
    writer.release()

    # Log video stats to docker logs for diagnostic visibility.
    try:
        file_size = os.path.getsize(video_path) if os.path.isfile(video_path) else 0
    except Exception:
        file_size = 0
    print(
        f"[sim_server] video: frames={grab_state['frames']} "
        f"bytes={file_size} path={video_path}",
        flush=True,
    )

    return {
        "ok": error is None,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "error": error,
        "elapsed_s": round(elapsed, 3),
        "script_path": script_path,
        "video_path": video_path,
        "video_frame_count": grab_state["frames"],
        "video_bytes": file_size,
    }


TOOLS = {
    "get_stage_info": _impl_get_stage_info,
    "query_stage": _impl_query_stage,
    "add_reference_to_stage": _impl_add_reference_to_stage,
    "set_transform": _impl_set_transform,
    "save_stage": _impl_save_stage,
    "search_assets": _impl_search_assets,
    "capture_viewport": _impl_capture_viewport,
    "add_light": _impl_add_light,
    "get_prim_bounds": _impl_get_prim_bounds,
    "delete_prim": _impl_delete_prim,
    "bind_material": _impl_bind_material,
    "run_python": _impl_run_python,
    "load_stage": _impl_load_stage,
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
            # 600s cap accommodates long-running run_python scripts that
            # step the sim through a task. Other RPCs return in well under
            # a second; the loose cap doesn't hurt them.
            result = fut.result(timeout=600.0)
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
