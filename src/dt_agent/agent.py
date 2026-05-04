"""
agent.py — Phase 2.5 orchestration: GPT-5.3-codex driving sim_server + VLM.

The agent is a single while-loop:

  1. Send the goal (and accumulated tool outputs) to GPT-5.3-codex via the
     Responses API on inference-api.nvidia.com.
  2. Read the model's response. If it contains tool calls, execute each
     against either the sim_server HTTP RPC (localhost:8765) or the VLM
     wrapper (`dt_agent.vlm.observe`).
  3. Feed the tool outputs back as the next turn's input.
  4. Stop when the model returns text without tool calls (it's done) or
     after `max_iterations`.

Every iteration is appended to a JSONL trace at output/agent_traces/.
Server-side state via `previous_response_id` keeps each turn's request
small — we only send the new function_call_output items.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from openai import OpenAI

from dt_agent.vlm import observe as vlm_observe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.environ.get(
    "NV_LLM_BASE_URL", "https://inference-api.nvidia.com/v1"
)
LLM_MODEL = os.environ.get("NV_LLM_MODEL", "openai/openai/gpt-5.3-codex")
SIM_BASE_URL = os.environ.get("DT_AGENT_SIM_URL", "http://localhost:8765")

USD_SEARCH_BASE_URL = "https://search.simready.omniverse.nvidia.com"
# Default to Isaac 5.1 assets so results are compatible with the running container.
USD_SEARCH_DEFAULT_PATH = None  # search all versions; 6.0 assets work in 5.1

# Repo root, used to translate container-side capture paths to host paths
# so the VLM wrapper (running on host) can read the PNG.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRACE_DIR = _PROJECT_ROOT / "output" / "agent_traces"


SYSTEM_PROMPT = """You are a digital-twin authoring agent for NVIDIA Isaac Sim.
You receive a natural-language goal describing a desired scene, and you have
tools to inspect, edit, and observe the scene to make it match the goal.

Workflow:
1. Survey: call `query_stage` to see what's already in the scene.
2. Plan: identify the prims you need with their target paths and transforms.
3. Build: use `create_primitive`, `add_reference_to_stage`, and `set_transform`
   to compose the scene. Use `search_assets_ai` to find NVIDIA library USDs by
   natural language description (e.g. "UR10 robot arm", "conveyor belt",
   "lab workbench") — pass the `url` field directly to `add_reference_to_stage`.
   Fall back to `search_assets` only for locally mounted custom assets.
4. Validate: call `observe(intent)` after substantive edits. The vision model
   returns `{intent_satisfied, observed, issues, correction_hint}`. If the
   observation contradicts what `query_stage` reports, trust `query_stage` for
   ground truth — the VLM may misidentify visually ambiguous geometry.
5. Iterate: when `intent_satisfied` is false, address the `issues` list one
   at a time. Re-observe between meaningful edits, not every primitive.
6. Save: when satisfied, `save_stage` to the requested file path and reply
   with a brief plain-text summary (no tool calls) to signal completion.

Conventions:
- Z-up world. Units are meters.
- USD Cube primitives have an authored size of 2.0 (span -1..+1 each axis).
  scale=[w/2, d/2, h/2] gives a box of dimensions w x d x h meters.
- Use prim paths under `/World/<your_subtree>/<name>`.
- Transform args: translate (xyz meters), rotate (xyz Euler degrees), scale.
- Save USDs to `/workspace/dt-agent/output/<name>.usda` so they appear on the
  host filesystem.

Respond with tool calls until the goal is achieved. Only emit plain text
(no tool calls) once you're done — that text is what gets returned to the
human who launched you."""


# ---------------------------------------------------------------------------
# Tool definitions exposed to the LLM (Responses API "function" tool shape)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
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
        "name": "create_primitive",
        "description": "Create a USD primitive at prim_path. Common types: Xform (empty group), Cube, Sphere, Cylinder, Cone, Capsule, Plane.",
        "parameters": {
            "type": "object",
            "properties": {
                "prim_path": {"type": "string"},
                "prim_type": {
                    "type": "string",
                    "description": "USD geometry type. Default 'Xform'.",
                },
            },
            "required": ["prim_path"],
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
        "name": "observe",
        "description": "Render the scene from the fixed observation camera and ask the VLM whether the rendered frame matches `intent`. Returns {intent_satisfied, observed, issues, correction_hint}. Slower than RPC tools (~2-5s); call after substantive edits, not after every primitive.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What the scene should look like, in plain English. Mention specific component names from the goal (e.g. 'UR10e', 'conveyor', 'three microplates') so the VLM can match them.",
                }
            },
            "required": ["intent"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------


def _sim_rpc(tool: str, **args) -> Any:
    """Call the sim_server RPC at localhost:8765."""
    req = urllib.request.Request(
        f"{SIM_BASE_URL}/rpc",
        data=json.dumps({"tool": tool, "args": args}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
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


def _exec_observe(intent: str) -> dict:
    """Composite tool: capture from the default observation camera, then
    send the resulting PNG to the VLM with the given intent."""
    cap = _sim_rpc("capture_viewport")
    if not cap.get("ok"):
        return {"error": f"capture_viewport failed: {cap}"}
    container_path = cap["file_path"]
    host_path = _container_to_host_path(container_path)
    obs = vlm_observe(host_path, intent)
    return obs.model_dump()


_TOOL_EXECUTORS = {
    "get_stage_info": lambda **kw: _sim_rpc("get_stage_info", **kw),
    "query_stage": lambda **kw: _sim_rpc("query_stage", **kw),
    "create_primitive": lambda **kw: _sim_rpc("create_primitive", **kw),
    "add_reference_to_stage": lambda **kw: _sim_rpc("add_reference_to_stage", **kw),
    "set_transform": lambda **kw: _sim_rpc("set_transform", **kw),
    "save_stage": lambda **kw: _sim_rpc("save_stage", **kw),
    "search_assets_ai": lambda **kw: _exec_search_assets_ai(**kw),
    "search_assets": lambda **kw: _sim_rpc("search_assets", **kw),
    "observe": lambda **kw: _exec_observe(**kw),
}


def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return a JSON-encoded string (what the Responses
    API expects as function_call_output.output). Exceptions are caught and
    surfaced to the model so it can self-correct."""
    if name not in _TOOL_EXECUTORS:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = _TOOL_EXECUTORS[name](**args)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def _make_tracer(trace_path: Path):
    def trace(event: str, **data):
        rec = {"t": time.time(), "event": event, **data}
        with trace_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    return trace


def _extract_text(response) -> str:
    """Pull the message text out of a Responses API response. The SDK
    provides `output_text` as a convenience but fall back to walking
    output items if that isn't populated."""
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                if t:
                    parts.append(t)
    return "\n".join(parts)


def run(goal: str, max_iterations: int = 30, *, log: bool = True) -> int:
    """Run the agent loop until the goal is satisfied or max_iterations elapse.

    Returns:
        0 on clean termination (model emitted text without tool calls).
        1 on iteration cap or unrecoverable error.
    """
    api_key = os.environ.get("NV_API_KEY")
    if not api_key:
        print("ERROR: NV_API_KEY not set", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)

    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = _TRACE_DIR / f"trace_{int(time.time() * 1000)}.jsonl"
    trace = _make_tracer(trace_path)

    if log:
        print(f"[agent] goal:  {goal}", file=sys.stderr)
        print(f"[agent] LLM:   {LLM_MODEL} via {LLM_BASE_URL}", file=sys.stderr)
        print(f"[agent] sim:   {SIM_BASE_URL}", file=sys.stderr)
        print(f"[agent] trace: {trace_path}", file=sys.stderr)

    trace("start", goal=goal, model=LLM_MODEL, sim=SIM_BASE_URL)

    # NV-internal proxy rejects `previous_response_id` (Zero Data Retention).
    # Maintain the full conversation history client-side and resend it as
    # `input` every turn. `instructions=` is re-sent every call too.
    history: list[dict[str, Any]] = [{"role": "user", "content": goal}]

    for iteration in range(1, max_iterations + 1):
        if log:
            print(f"\n[agent] -- iteration {iteration}/{max_iterations} --", file=sys.stderr)

        try:
            response = client.responses.create(
                model=LLM_MODEL,
                instructions=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                input=history,
            )
        except Exception as e:
            trace("llm_error", iteration=iteration, error=str(e))
            print(f"[agent] LLM call failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

        trace("llm_response", iteration=iteration, response_id=response.id)

        # Echo every output item back into history so the next turn carries
        # the full conversation. `model_dump()` on each item produces the
        # right shape for the Responses API to accept as input.
        tool_calls = []
        for item in response.output or []:
            try:
                history.append(item.model_dump())
            except Exception:
                # If the item type isn't a Pydantic model for some reason,
                # fall back to a best-effort dict so we don't break the loop.
                history.append({"type": getattr(item, "type", "unknown")})
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(item)

        if not tool_calls:
            final_text = _extract_text(response) or "(no text returned)"
            if log:
                print(f"\n[agent] DONE\n{final_text}", file=sys.stderr)
            trace("done", iteration=iteration, final_text=final_text)
            print(final_text)
            return 0

        # Execute each tool call and append the outputs to history.
        for call in tool_calls:
            args = json.loads(call.arguments) if call.arguments else {}
            if log:
                args_repr = json.dumps(args)
                if len(args_repr) > 160:
                    args_repr = args_repr[:160] + "..."
                print(f"[agent]   -> {call.name}({args_repr})", file=sys.stderr)
            output = _execute_tool(call.name, args)
            if log:
                trim = output if len(output) <= 240 else output[:240] + "..."
                print(f"[agent]      = {trim}", file=sys.stderr)
            trace(
                "tool_call",
                iteration=iteration,
                name=call.name,
                args=args,
                output=output,
            )
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

    if log:
        print(
            f"\n[agent] hit iteration cap ({max_iterations}); stopping.",
            file=sys.stderr,
        )
    trace("max_iter_reached")
    return 1
