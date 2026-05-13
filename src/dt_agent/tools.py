"""
tools.py — Tool surface exposed to the agent loop.

Everything in here describes *what the agent can do*, independent of *what
it's being asked to do*. The scene-authoring SYSTEM_PROMPT and the loop's
guard configuration live in agent.py and loop.py respectively; this module
just provides the building blocks they assemble.

Exports:
- TOOL_DEFINITIONS — Chat Completions tool list ready to pass to OpenAI SDK
- TOOL_EXECUTORS   — dispatch map: name -> callable(**args)
- EDIT_TOOLS       — names that count as stage edits (for the cadence gate)
- PROGRESS_TOOLS   — names that count as real progress (for auto-extend)
- EDIT_CADENCE     — max edits without an intervening observe()
- sim_rpc          — exposed for future tools (e.g. robot-tasks) that need
                     to hit sim_server endpoints directly
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dt_agent.vlm import observe as vlm_observe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIM_BASE_URL = os.environ.get("DT_AGENT_SIM_URL", "http://localhost:8765")

USD_SEARCH_BASE_URL = "https://search.simready.omniverse.nvidia.com"
# Default to Isaac 5.1 assets so results are compatible with the running container.
USD_SEARCH_DEFAULT_PATH = None  # search all versions; 6.0 assets work in 5.1

# Repo root, used to translate container-side capture paths to host paths
# so the VLM wrapper (running on host) can read the PNG.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# RPC + path helpers
# ---------------------------------------------------------------------------


def sim_rpc(tool: str, **args) -> Any:
    """Call the sim_server RPC at localhost:8765."""
    req = urllib.request.Request(
        f"{SIM_BASE_URL}/rpc",
        data=json.dumps({"tool": tool, "args": args}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # The server puts {"error": "<ExceptionType>: <msg>"} in 4xx/5xx bodies;
        # urlopen raises before we'd otherwise read it. Pull the real error out
        # so the agent trace shows the actual Kit/USD exception, not a bare
        # "HTTP 500 Internal Server Error".
        try:
            err_body = json.loads(e.read())
            detail = err_body.get("error") or f"HTTP {e.code} (empty body)"
        except Exception:
            detail = f"HTTP {e.code} {e.reason}"
        raise RuntimeError(f"{tool}: {detail}") from None
    if "error" in body:
        raise RuntimeError(f"{tool}: {body['error']}")
    return body["result"]


def _container_to_host_path(container_path: str) -> str:
    """Translate a sim-side path under /workspace/dt-agent/output/ into the
    corresponding host path under <repo>/output/, since vlm_observe runs on
    the host and reads the PNG file directly."""
    container_prefix = "/workspace/dt-agent/output/"
    if container_path.startswith(container_prefix):
        rel = container_path[len(container_prefix):]
        return str(_PROJECT_ROOT / "output" / rel)
    return container_path


# ---------------------------------------------------------------------------
# Material catalog (baked subset of Isaac Sim's MDL library)
# ---------------------------------------------------------------------------

_MATERIAL_BASE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/6.0/Isaac/Materials"
)

_MATERIAL_CATALOG: list[dict] = [
    # Base/Metals
    {"name": "Aluminum_Anodized",       "url": f"{_MATERIAL_BASE}/Base/Metals/Aluminum_Anodized.mdl",       "tags": ["aluminum", "metal", "anodized", "silver", "smooth"]},
    {"name": "Aluminum_Anodized_Black", "url": f"{_MATERIAL_BASE}/Base/Metals/Aluminum_Anodized_Black.mdl", "tags": ["aluminum", "metal", "anodized", "black", "dark"]},
    {"name": "Aluminum_Cast",           "url": f"{_MATERIAL_BASE}/Base/Metals/Aluminum_Cast.mdl",           "tags": ["aluminum", "metal", "cast", "rough", "grey"]},
    {"name": "Brass",                   "url": f"{_MATERIAL_BASE}/Base/Metals/Brass.mdl",                   "tags": ["brass", "metal", "gold", "yellow", "warm"]},
    {"name": "Brushed_Antique_Copper",  "url": f"{_MATERIAL_BASE}/Base/Metals/Brushed_Antique_Copper.mdl",  "tags": ["copper", "metal", "brushed", "antique", "orange", "warm"]},
    {"name": "Copper",                  "url": f"{_MATERIAL_BASE}/Base/Metals/Copper.mdl",                  "tags": ["copper", "metal", "orange", "shiny"]},
    {"name": "Steel_Stainless",         "url": f"{_MATERIAL_BASE}/Base/Metals/Steel_Stainless.mdl",         "tags": ["steel", "stainless", "metal", "silver", "grey", "industrial"]},
    # Base/Plastics
    {"name": "Plastic_ABS",             "url": f"{_MATERIAL_BASE}/Base/Plastics/Plastic_ABS.mdl",           "tags": ["plastic", "abs", "polymer", "hard", "matte"]},
    {"name": "Rubber_Smooth",           "url": f"{_MATERIAL_BASE}/Base/Plastics/Rubber_Smooth.mdl",         "tags": ["rubber", "smooth", "elastic", "soft", "black"]},
    {"name": "Rubber_Textured",         "url": f"{_MATERIAL_BASE}/Base/Plastics/Rubber_Textured.mdl",       "tags": ["rubber", "textured", "elastic", "grip", "black"]},
    {"name": "Vinyl",                   "url": f"{_MATERIAL_BASE}/Base/Plastics/Vinyl.mdl",                 "tags": ["vinyl", "plastic", "smooth", "floor", "covering"]},
    # Base/Masonry
    {"name": "Brick_Pavers",            "url": f"{_MATERIAL_BASE}/Base/Masonry/Brick_Pavers.mdl",           "tags": ["brick", "masonry", "pavers", "stone", "floor", "wall", "concrete"]},
    # Base/Carpet
    {"name": "Carpet_Diamond_Yellow",   "url": f"{_MATERIAL_BASE}/Base/Carpet/Carpet_Diamond_Yellow.mdl",   "tags": ["carpet", "fabric", "yellow", "floor", "soft"]},
    # vMaterials_2/Metal
    {"name": "Aluminum_Brushed",        "url": f"{_MATERIAL_BASE}/vMaterials_2/Metal/Aluminum_Brushed.mdl", "tags": ["aluminum", "metal", "brushed", "silver", "grey"]},
    {"name": "Aluminum_Scratched",      "url": f"{_MATERIAL_BASE}/vMaterials_2/Metal/Aluminum_Scratched.mdl","tags": ["aluminum", "metal", "scratched", "worn", "aged"]},
    {"name": "Stainless_Steel",         "url": f"{_MATERIAL_BASE}/vMaterials_2/Metal/Stainless_Steel.mdl",  "tags": ["steel", "stainless", "metal", "silver", "shiny", "industrial"]},
]


def _exec_search_materials(query: str, limit: int = 5) -> dict:
    """Keyword search over the baked Isaac Sim material catalog."""
    words = query.lower().split()
    scored = []
    for mat in _MATERIAL_CATALOG:
        haystack = mat["name"].lower().replace("_", " ") + " " + " ".join(mat["tags"])
        score = sum(haystack.count(w) for w in words)
        if score > 0:
            scored.append((score, mat))
    scored.sort(key=lambda x: x[0], reverse=True)
    matches = [
        {"name": m["name"], "url": m["url"], "tags": m["tags"]}
        for _, m in scored[:limit]
    ]
    if not matches:
        # Return full catalog so the agent can still choose
        matches = [{"name": m["name"], "url": m["url"], "tags": m["tags"]} for m in _MATERIAL_CATALOG]
    return {"matches": matches, "count": len(matches)}


def _exec_search_assets_ai(
    description: str,
    limit: int = 10,
    search_path: str | None = USD_SEARCH_DEFAULT_PATH,
) -> dict:
    """Call the USD Search API and return asset URLs ready for add_reference_to_stage."""
    api_key = os.environ.get("NV_API_KEY")
    if not api_key:
        return {"error": "NV_API_KEY not set"}

    import urllib.parse
    params: dict[str, Any] = {
        "description": description,
        "file_extension_include": "usd",
        "limit": limit,
        "return_metadata": "false",
    }
    if search_path:
        params["search_path"] = search_path

    url = f"{USD_SEARCH_BASE_URL}/search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"USD Search HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    items = body if isinstance(body, list) else []
    matches = [
        {
            "url": item.get("url", ""),
            "score": round(float(item.get("score", 0)), 4),
            "name": item.get("url", "").rsplit("/", 1)[-1],
        }
        for item in items
    ]
    return {"matches": matches, "count": len(matches)}


def _exec_observe(
    intent: str,
    eye: list | None = None,
    target: list | None = None,
) -> dict:
    """Composite tool: capture from the observation camera, then send the
    resulting PNG to the VLM with the given intent. Pass eye/target to
    reposition the camera for large scenes."""
    cap_args: dict[str, Any] = {}
    if eye is not None:
        cap_args["eye"] = eye
    if target is not None:
        cap_args["target"] = target
    cap = sim_rpc("capture_viewport", **cap_args)
    if not cap.get("ok"):
        return {"error": f"capture_viewport failed: {cap}"}
    container_path = cap["file_path"]
    host_path = _container_to_host_path(container_path)
    obs = vlm_observe(host_path, intent)
    return obs.model_dump()


# ---------------------------------------------------------------------------
# Tool definitions exposed to the LLM (Responses API "function" tool shape)
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS_RAW: list[dict] = [
    {
        "type": "function",
        "name": "get_stage_info",
        "description": "Return the loaded USD stage URL and total prim count. Quick health check.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "query_stage",
        "description": "List child prims under prim_path with their type and world translate. Use to survey the scene before editing or to verify edits afterward.",
        "parameters": {
            "type": "object",
            "properties": {
                "prim_path": {
                    "type": "string",
                    "description": "USD prim path. Use '/' for full traversal.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Levels to descend. 1=just children, 3=default, higher for deeper.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "add_reference_to_stage",
        "description": "Bring an external USD asset into the stage as a reference at prim_path. If prim_path doesn't exist, an Xform container is created. Use URLs from search_assets.",
        "parameters": {
            "type": "object",
            "properties": {
                "usd_path": {
                    "type": "string",
                    "description": "URL or local file path of the USD to reference.",
                },
                "prim_path": {"type": "string"},
            },
            "required": ["usd_path", "prim_path"],
        },
    },
    {
        "type": "function",
        "name": "set_transform",
        "description": "Set translate (xyz meters), rotate (xyz Euler degrees), and/or scale on an existing prim. Any axis may be omitted; pass only what you want to change.",
        "parameters": {
            "type": "object",
            "properties": {
                "prim_path": {"type": "string"},
                "translate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "rotate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "scale": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["prim_path"],
        },
    },
    {
        "type": "function",
        "name": "save_stage",
        "description": "Save the current stage to file_path inside the container. Use /workspace/dt-agent/output/<name>.usda so the file appears on the host.",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "type": "function",
        "name": "search_assets_ai",
        "description": "Search NVIDIA's Isaac/SimReady asset library by natural language description via the USD Search API. Returns USD URLs ready to pass directly to add_reference_to_stage. Defaults to Isaac 5.1 assets (compatible with the running container). Prefer this over search_assets for any NVIDIA library asset.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the desired asset, e.g. 'UR10 robot arm', 'industrial conveyor belt', 'lab workbench', 'microplate'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results. Default 10.",
                },
                "search_path": {
                    "type": "string",
                    "description": "Restrict search to a CDN path prefix. Leave unset to search all Isaac versions.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "type": "function",
        "name": "search_assets",
        "description": "Search the curated NVIDIA OpenUSD CDN catalog AND in-image filesystem for USDs matching `query`. Returns matches with path/url, source, name, description, verified flag.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Substring to match (e.g. 'ur10', 'franka', 'warehouse').",
                },
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "add_light",
        "description": "Create a USD light prim. Required for interior lighting in enclosed spaces (rooms, warehouses) — the default DomeLight is blocked by walls and ceilings. Use SphereLight for point/ceiling lights, RectLight for overhead panels, DiskLight for circular ceiling fixtures.",
        "parameters": {
            "type": "object",
            "properties": {
                "prim_path": {"type": "string"},
                "light_type": {
                    "type": "string",
                    "enum": ["SphereLight", "RectLight", "DiskLight", "CylinderLight"],
                    "description": "USD light type. Default SphereLight.",
                },
                "intensity": {
                    "type": "number",
                    "description": "Photometric intensity. 3000–10000 for warehouse-scale spaces. Default 3000.",
                },
                "translate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Light position in metres (xyz).",
                },
                "color": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "RGB color, each 0–1. Default white.",
                },
                "radius": {"type": "number", "description": "Radius in metres for SphereLight/DiskLight."},
                "width": {"type": "number", "description": "Width in metres for RectLight."},
                "height": {"type": "number", "description": "Height in metres for RectLight."},
            },
            "required": ["prim_path"],
        },
    },
    {
        "type": "function",
        "name": "get_prim_bounds",
        "description": "Return the world-space axis-aligned bounding box (min, max, size in metres) of a prim. Use after add_reference_to_stage to measure a loaded asset before tiling it.",
        "parameters": {
            "type": "object",
            "properties": {
                "prim_path": {"type": "string"},
            },
            "required": ["prim_path"],
        },
    },
    {
        "type": "function",
        "name": "delete_prim",
        "description": "Remove a prim and all its descendants from the stage. Use to clean up probe assets after you've measured them with get_prim_bounds, or to undo a misplaced reference. Probes left on the stage will appear in observe() captures and confuse the VLM.",
        "parameters": {
            "type": "object",
            "properties": {
                "prim_path": {"type": "string"},
            },
            "required": ["prim_path"],
        },
    },
    {
        "type": "function",
        "name": "search_materials",
        "description": "Search the Isaac Sim built-in material catalog by keyword. Returns MDL URLs ready to pass to bind_material. Available categories: metals (aluminum, brass, copper, stainless steel), plastics (ABS, rubber, vinyl), masonry (brick), carpet. If no query matches, returns the full catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the desired material, e.g. 'stainless steel', 'brushed aluminum', 'rubber grip', 'concrete floor'.",
                },
                "limit": {"type": "integer", "description": "Max results. Default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "bind_material",
        "description": "Apply an MDL material (from search_materials) to a prim. Creates a material prim under /World/Looks/ and binds it. Re-binding the same material URL to multiple prims reuses the same material prim.",
        "parameters": {
            "type": "object",
            "properties": {
                "prim_path": {"type": "string", "description": "Path of the prim to receive the material."},
                "material_url": {"type": "string", "description": "HTTPS MDL URL from search_materials."},
            },
            "required": ["prim_path", "material_url"],
        },
    },
    {
        "type": "function",
        "name": "observe",
        "description": "Render the scene and ask the VLM whether it matches `intent`. Returns {intent_satisfied, observed, issues, correction_hint}. Slower than RPC tools; call after substantive edits. For large scenes pass eye/target so the camera frames the whole scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What the scene should look like. Mention specific component names so the VLM can match them.",
                },
                "eye": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Camera eye position in metres (xyz). Default (3,3,2) suits ~2m scenes. For a 20×40m warehouse try (-30,-30,20).",
                },
                "target": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Camera look-at point in metres (xyz). Default (0.6,0,0.45). For a warehouse centred at origin try (0,0,5).",
                },
            },
            "required": ["intent"],
        },
    },
]

# Wrap into Chat Completions tool format: {type, function: {name, description, parameters}}
TOOL_DEFINITIONS: list[dict] = [
    {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
    for t in _TOOL_DEFINITIONS_RAW
]


TOOL_EXECUTORS = {
    "get_stage_info": lambda **kw: sim_rpc("get_stage_info", **kw),
    "query_stage": lambda **kw: sim_rpc("query_stage", **kw),
    "add_reference_to_stage": lambda **kw: sim_rpc("add_reference_to_stage", **kw),
    "set_transform": lambda **kw: sim_rpc("set_transform", **kw),
    "save_stage": lambda **kw: sim_rpc("save_stage", **kw),
    "search_assets_ai": lambda **kw: _exec_search_assets_ai(**kw),
    "search_assets": lambda **kw: sim_rpc("search_assets", **kw),
    "search_materials": lambda **kw: _exec_search_materials(**kw),
    "bind_material": lambda **kw: sim_rpc("bind_material", **kw),
    "get_prim_bounds": lambda **kw: sim_rpc("get_prim_bounds", **kw),
    "add_light": lambda **kw: sim_rpc("add_light", **kw),
    "delete_prim": lambda **kw: sim_rpc("delete_prim", **kw),
    "observe": lambda **kw: _exec_observe(**kw),
}

# Edit cadence: tools that mutate the stage. After EDIT_CADENCE such calls
# without an intervening observe(), further edits are blocked until the model
# observes — prevents "build everything, then observe once at the end."
EDIT_TOOLS = {"add_reference_to_stage", "set_transform", "add_light", "bind_material"}
EDIT_CADENCE = 8

# Tools whose successful execution counts as "real progress" toward the build.
# Used by the auto-extend logic in run_turn to decide whether to keep going
# past the soft iteration cap.
PROGRESS_TOOLS = {
    "add_reference_to_stage",
    "set_transform",
    "add_light",
    "bind_material",
    "delete_prim",
    "save_stage",
}
