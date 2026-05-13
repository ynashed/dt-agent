"""
tools_tasks.py — Tool surface for the robot-task author agent.

Composed of reused tools from tools.py (scene introspection, asset search,
observe) plus new ones for writing and running Python task scripts in the
Kit container.

The task agent operates on a pre-loaded scene — load_stage is called by
chat_tasks.py at startup, not by the agent itself.

Exports:
- TOOL_DEFINITIONS    — Chat Completions tool list
- TOOL_EXECUTORS      — dispatch map: name -> callable(**args)
- EDIT_TOOLS          — names that count as "edits" (cadence gate)
- PROGRESS_TOOLS      — names that count as real progress (auto-extend)
- EDIT_CADENCE        — max edits without an intervening observe()
"""
from __future__ import annotations

import os
from pathlib import Path

from dt_agent.tools import (
    _exec_observe,
    _exec_search_assets_ai,
    sim_rpc,
)

# Same path the sim_server sees inside the container — the docker-compose
# mount makes /workspace/dt-agent/output/ on the host equal to inside the
# container, so a single absolute path works on both sides.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "output" / "scripts"


def _exec_write_script(path: str, contents: str) -> dict:
    """Write a Python script to the shared volume so run_python can exec it.

    `path` must be under /workspace/dt-agent/output/scripts/ so the sim
    container can read it. The host-side equivalent is <repo>/output/scripts/.
    """
    prefix = "/workspace/dt-agent/output/scripts/"
    if not path.startswith(prefix):
        return {
            "ok": False,
            "error": (
                f"script path must start with {prefix} so the sim container "
                f"can read it (got: {path})"
            ),
        }
    rel = path[len(prefix):]
    host_path = _SCRIPTS_DIR / rel
    try:
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_text(contents)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {
        "ok": True,
        "script_path": path,
        "bytes_written": len(contents.encode("utf-8")),
    }


def _exec_run_python(script_path: str) -> dict:
    """Pass-through to the sim_server's run_python RPC."""
    return sim_rpc("run_python", script_path=script_path)


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
        "description": "List child prims under prim_path with their type and world translate. Use to find the robot's prim path and the prim paths of relevant objects in the loaded scene.",
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
        "name": "get_prim_bounds",
        "description": "Return the world-space axis-aligned bounding box (min, max, size in metres) of a prim. Use to confirm an object's position before scripting a motion toward it.",
        "parameters": {
            "type": "object",
            "properties": {"prim_path": {"type": "string"}},
            "required": ["prim_path"],
        },
    },
    {
        "type": "function",
        "name": "search_assets",
        "description": "Search the curated NVIDIA OpenUSD CDN catalog AND in-image filesystem for USDs matching `query`. Use only if the task requires bringing in a new asset (rare — the scene is usually pre-authored).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "search_assets_ai",
        "description": "Search NVIDIA's Isaac/SimReady asset library by natural language description via the USD Search API. Use only if you need to introduce a new asset to the scene; usually unnecessary for task scripting.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "limit": {"type": "integer"},
                "search_path": {"type": "string"},
            },
            "required": ["description"],
        },
    },
    {
        "type": "function",
        "name": "write_script",
        "description": "Write a Python script to the shared volume. Path must start with /workspace/dt-agent/output/scripts/ so run_python can read it. The script can import omni, pxr, and use sim_app.update() / world.step() to step the simulation. Use versioned names (foo_v1.py, foo_v2.py) when iterating.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path under /workspace/dt-agent/output/scripts/, e.g. '/workspace/dt-agent/output/scripts/pick_cube_v1.py'.",
                },
                "contents": {
                    "type": "string",
                    "description": "Full Python source for the script.",
                },
            },
            "required": ["path", "contents"],
        },
    },
    {
        "type": "function",
        "name": "run_python",
        "description": "Execute a script file via the sim_server RPC. Runs on Kit's main thread with full omni/pxr access. Returns {ok, stdout, stderr, error, elapsed_s}. A non-null error field is a Python exception with traceback.",
        "parameters": {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": "Container-side path to the script, as written by write_script.",
                },
            },
            "required": ["script_path"],
        },
    },
    {
        "type": "function",
        "name": "save_stage",
        "description": "Save the current stage to file_path inside the container. Use to persist the post-task scene state, e.g. /workspace/dt-agent/output/<name>_after_task.usda.",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "type": "function",
        "name": "observe",
        "description": "Render the scene from the observation camera and ask the VLM whether it matches `intent`. Returns {intent_satisfied, observed, issues, correction_hint}. Call after run_python finishes to verify the task outcome. For large scenes pass eye/target so the camera frames the relevant area.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What the final scene should look like (e.g. 'the red cube is inside the blue container').",
                },
                "eye": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "target": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["intent"],
        },
    },
]


# Wrap into Chat Completions tool format.
TOOL_DEFINITIONS: list[dict] = [
    {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
    for t in _TOOL_DEFINITIONS_RAW
]


TOOL_EXECUTORS = {
    "get_stage_info": lambda **kw: sim_rpc("get_stage_info", **kw),
    "query_stage": lambda **kw: sim_rpc("query_stage", **kw),
    "get_prim_bounds": lambda **kw: sim_rpc("get_prim_bounds", **kw),
    "search_assets": lambda **kw: sim_rpc("search_assets", **kw),
    "search_assets_ai": lambda **kw: _exec_search_assets_ai(**kw),
    "write_script": lambda **kw: _exec_write_script(**kw),
    "run_python": lambda **kw: _exec_run_python(**kw),
    "save_stage": lambda **kw: sim_rpc("save_stage", **kw),
    "observe": lambda **kw: _exec_observe(**kw),
}


# Edit cadence: after this many run_python calls without an intervening
# observe, block further runs. Tighter than the authoring cadence because
# each task-script run is more substantive than a single stage edit.
EDIT_TOOLS = {"run_python"}
EDIT_CADENCE = 3

# Progress tools — what counts as forward motion for the auto-extend logic.
PROGRESS_TOOLS = {"write_script", "run_python", "save_stage"}
