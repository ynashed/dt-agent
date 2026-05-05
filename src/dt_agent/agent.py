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
LLM_MODEL = os.environ.get("NV_LLM_MODEL", "aws/anthropic/bedrock-claude-opus-4-7")
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
3. Build: use `create_primitive`, `add_reference_to_stage`, `set_transform`,
   and `get_prim_bounds` to compose the scene.
   - For every named component (robot, wall, floor, conveyor, etc.) call
     `search_assets_ai` first. If it returns relevant results, you MUST use
     `add_reference_to_stage` with one of those URLs — do NOT substitute
     Cube/Sphere primitives.
   - For modular/tiled assets (wall panels, floor tiles): load ONE instance,
     call `get_prim_bounds` to measure it, calculate tile positions, then
     tile with `add_reference_to_stage` + `set_transform`.
   - Only use Cube/Sphere/Cylinder primitives for components where search
     returned no usable asset.
4. Validate: call `observe(intent)` after substantive edits. The vision model
   returns `{intent_satisfied, observed, issues, correction_hint}`. If the
   observation contradicts what `query_stage` reports, trust `query_stage` for
   ground truth — the VLM may misidentify visually ambiguous geometry.
   A completely black or near-black image means no light reaches the camera —
   do NOT save or declare done. For enclosed spaces add interior lights with
   `add_light` before observing again.
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
- Interior lighting: the default DomeLight added by `capture_viewport` is
  blocked by walls and ceilings. For any enclosed space, use `add_light` to
  place SphereLight or RectLight prims inside (intensity 3000–10000 for
  warehouse scale). Place lights before the first `observe` call.
- Camera for large scenes: the default eye (3,3,2) is designed for ~2m
  workcells. For scenes >5m, pass eye and target to `observe`. A 20×40m
  warehouse centred at the origin: eye=[-30,-30,20], target=[0,0,5].
  Scale the distance proportionally for other sizes.
- Do NOT declare the task done if `observe` returns an error or a black image.
  Diagnose and fix (add lights, reposition camera, fix geometry) then re-observe.
- Do NOT call `save_stage` while the last `observe()` returned
  `intent_satisfied=false` — the framework will block it. Fix all listed
  issues, call `observe()` again, and only then save.

Tool batching: always call independent tools in parallel within a single
response — never wait for one result before issuing the next unrelated call.
Examples of things that should be one response:
- All `search_assets_ai` lookups for a scene (walls, floors, ceiling, etc.)
- All `add_reference_to_stage` calls once you have the URLs
- All `set_transform` calls once you know the positions
- All `add_light` calls
The only time you must wait is when a result feeds the next call
(e.g. `get_prim_bounds` needs the prim to exist first).

Conversational / follow-up turns:
- When the user asks to "check", "look", or "see" what the scene looks like,
  call `observe` immediately — do not describe what you plan to do.
- When the user confirms a proposed plan ("ok", "yes", "go ahead", "proceed",
  "sure"), execute the plan immediately with tool calls — do not restate the
  plan in text.
- Never return an empty response. If you have nothing to say after a tool call
  chain, summarize what you did in one sentence.

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
TOOL_DEFINITIONS = [
    {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
    for t in TOOL_DEFINITIONS
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
    cap = _sim_rpc("capture_viewport", **cap_args)
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
    "get_prim_bounds": lambda **kw: _sim_rpc("get_prim_bounds", **kw),
    "add_light": lambda **kw: _sim_rpc("add_light", **kw),
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
    """Pull the assistant's text content from a Chat Completions response."""
    if response.choices:
        return response.choices[0].message.content or ""
    return ""


def run_turn(
    message: str,
    history: "list[dict[str, Any]]",
    client: "OpenAI",
    trace,
    max_iterations: int = 30,
    *,
    log: bool = True,
    enforce_save: bool = False,
    enforce_observe: bool = False,
) -> tuple[str, bool]:
    """Execute one conversational turn against an existing history.

    Appends `message` as a user turn, runs the tool loop until the model
    emits text with no tool calls, and returns (response_text, hit_cap).

    `history` is mutated in-place — the caller owns it across turns.
    `enforce_save`: intercept any attempt to finish before save_stage succeeds.
    `enforce_observe`: intercept any attempt to finish before observe() is called.
    Both gates together prevent the model from hallucinating task completion.
    """
    history.append({"role": "user", "content": message})
    save_stage_succeeded = False
    stall_count = 0  # consecutive empty-response count; reset on any tool call
    last_vlm_issues: list[str] = []     # issues from the most recent observe() call
    last_vlm_satisfied: bool | None = None  # None = no observe yet this turn

    for iteration in range(1, max_iterations + 1):
        if log:
            print(f"\n[agent] -- iteration {iteration}/{max_iterations} --", file=sys.stderr)

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                tools=TOOL_DEFINITIONS,
                parallel_tool_calls=True,
                #temperature=0.2,
            )
        except Exception as e:
            trace("llm_error", iteration=iteration, error=str(e))
            print(f"[agent] LLM call failed: {type(e).__name__}: {e}", file=sys.stderr)
            return f"LLM call failed: {e}", False

        trace("llm_response", iteration=iteration, response_id=response.id)

        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        # Append assistant turn to history (include tool_calls array if present)
        msg_dict: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        history.append(msg_dict)

        if not tool_calls:
            # Gate 1: require at least one observe() before finishing a build task.
            if enforce_observe and last_vlm_satisfied is None:
                nudge = (
                    "You must call observe() to visually verify the scene before "
                    "declaring done. Call observe(intent=...) now with an appropriate "
                    "eye/target for the scene size."
                )
                history.append({"role": "user", "content": nudge})
                if log:
                    print("[agent]  NOTE: intercepted premature finish — observe() not yet called", file=sys.stderr)
                trace("observe_nudge", iteration=iteration)
                continue

            # Gate 2: require save_stage to succeed before finishing.
            if enforce_save and not save_stage_succeeded:
                nudge = (
                    "You declared the task done but save_stage was never "
                    "successfully called — the output file has not been written. "
                    "Call save_stage with the requested file path now."
                )
                history.append({"role": "user", "content": nudge})
                if log:
                    print("[agent]  NOTE: intercepted premature finish — save_stage not yet called", file=sys.stderr)
                trace("save_nudge", iteration=iteration)
                continue
            final_text = _extract_text(response)
            if not final_text:
                stall_count += 1
                if stall_count <= 2:
                    nudge = "Please continue — execute the next step or summarize what you just did."
                    history.append({"role": "user", "content": nudge})
                    if log:
                        print(f"[agent]  NOTE: empty response (stall {stall_count}/2), nudging", file=sys.stderr)
                    trace("stall_nudge", iteration=iteration, stall_count=stall_count)
                    continue
                trace("done", iteration=iteration, final_text="(stalled after 2 nudges)")
                return "(stalled — model returned empty responses)", False
            stall_count = 0
            trace("done", iteration=iteration, final_text=final_text)
            return final_text, False

        stall_count = 0  # any tool call means the model is working
        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments) if call.function.arguments else {}

            # Guard: block save_stage if the last observe() was not satisfied.
            if name == "save_stage" and last_vlm_satisfied is False:
                issues_str = "; ".join(last_vlm_issues) if last_vlm_issues else "see correction_hint"
                blocked_output = json.dumps({
                    "error": (
                        f"save_stage blocked: the last observe() returned "
                        f"intent_satisfied=false (issues: {issues_str}). "
                        "Fix the listed issues and call observe() again to "
                        "confirm before saving."
                    )
                })
                if log:
                    print(
                        f"[agent]  BLOCKED save_stage — last observe() not satisfied "
                        f"(issues: {issues_str})",
                        file=sys.stderr,
                    )
                trace("save_blocked", iteration=iteration, issues=last_vlm_issues)
                history.append({"role": "tool", "tool_call_id": call.id, "content": blocked_output})
                continue

            if log:
                args_repr = json.dumps(args)
                if len(args_repr) > 160:
                    args_repr = args_repr[:160] + "..."
                print(f"[agent]   -> {name}({args_repr})", file=sys.stderr)
            output = _execute_tool(name, args)

            # Track observe() outcomes so the save guard stays current.
            if name == "observe":
                try:
                    obs = json.loads(output)
                    last_vlm_satisfied = bool(obs.get("intent_satisfied"))
                    last_vlm_issues = obs.get("issues") or []
                except Exception:
                    pass

            if name == "save_stage":
                try:
                    if json.loads(output).get("ok"):
                        save_stage_succeeded = True
                        last_vlm_satisfied = None  # reset guard after a successful save
                except Exception:
                    pass
            if log:
                trim = output if len(output) <= 240 else output[:240] + "..."
                print(f"[agent]      = {trim}", file=sys.stderr)
            trace("tool_call", iteration=iteration, name=name, args=args, output=output)
            history.append({"role": "tool", "tool_call_id": call.id, "content": output})

    if log:
        print(f"\n[agent] hit iteration cap ({max_iterations}); stopping.", file=sys.stderr)
    trace("max_iter_reached")
    return f"(hit iteration cap of {max_iterations})", True


def run(goal: str, max_iterations: int = 30, *, log: bool = True) -> int:
    """One-shot entrypoint: run a single goal to completion and exit.

    Returns 0 on success, 1 on error or iteration cap.
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

    history: list[dict[str, Any]] = []
    final_text, hit_cap = run_turn(
        goal, history, client, trace,
        max_iterations=max_iterations, log=log,
        enforce_save=True, enforce_observe=True,
    )
    print(final_text)
    return 1 if hit_cap else 0
