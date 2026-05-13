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
from dt_agent.vlm import observe_video as _vlm_observe_video

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


# Container-side video paths translate to host paths the same way as captures.
_OUTPUT_PREFIX_CONTAINER = "/workspace/dt-agent/output/"
_OUTPUT_PREFIX_HOST = Path(__file__).resolve().parents[2] / "output"


def _container_to_host_path(container_path: str) -> str:
    if container_path.startswith(_OUTPUT_PREFIX_CONTAINER):
        rel = container_path[len(_OUTPUT_PREFIX_CONTAINER):]
        return str(_OUTPUT_PREFIX_HOST / rel)
    return container_path


def _exec_observe_video(video_path: str, intent: str) -> dict:
    """Run Cosmos Reason against the mp4 recorded by run_python. Translates
    the container-side path returned by run_python to the host path the VLM
    wrapper reads from."""
    host_path = _container_to_host_path(video_path)
    obs = _vlm_observe_video(host_path, intent)
    return obs.model_dump()


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
        "name": "set_transform",
        "description": "Set translate (xyz meters), rotate (xyz Euler degrees), and/or scale on an existing prim. Use for STATIC repositioning of scene objects (move a target cube into reach, reorient the robot base, etc.) — cheaper and clearer than writing a script for it. For motion over time, use run_python instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "prim_path": {"type": "string"},
                "translate": {
                    "type": "array", "items": {"type": "number"},
                    "minItems": 3, "maxItems": 3,
                },
                "rotate": {
                    "type": "array", "items": {"type": "number"},
                    "minItems": 3, "maxItems": 3,
                },
                "scale": {
                    "type": "array", "items": {"type": "number"},
                    "minItems": 3, "maxItems": 3,
                },
            },
            "required": ["prim_path"],
        },
    },
    {
        "type": "function",
        "name": "delete_prim",
        "description": "Remove a prim and all its descendants from the stage. Use to clear stale setup artifacts or remove a prim that's blocking the task. Cannot remove `/`, `/World`, or `''`.",
        "parameters": {
            "type": "object",
            "properties": {"prim_path": {"type": "string"}},
            "required": ["prim_path"],
        },
    },
    {
        "type": "function",
        "name": "add_reference_to_stage",
        "description": "Reference an external USD asset into the loaded scene. Use for setup: introducing target objects (e.g., a cube for pick-and-place) that aren't already in the authored scene. URLs come from search_assets / search_assets_ai.",
        "parameters": {
            "type": "object",
            "properties": {
                "usd_path": {"type": "string"},
                "prim_path": {"type": "string"},
            },
            "required": ["usd_path", "prim_path"],
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
        "name": "observe_video",
        "description": "Send the mp4 recorded by run_python to the VLM and ask whether the task succeeded. Prefer this over observe() for task verification — the video shows the whole trajectory, not just the final frame. Pass video_path from run_python's result.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Container-side path returned in run_python's `video_path` field.",
                },
                "intent": {
                    "type": "string",
                    "description": "What the task should have visibly accomplished (e.g. 'the robotic arm picked the red cube and placed it in the blue container').",
                },
            },
            "required": ["video_path", "intent"],
        },
    },
    {
        "type": "function",
        "name": "observe",
        "description": "Render a single still from the observation camera and ask the VLM what it sees. Use only for static checks (no motion involved); for task verification prefer observe_video which shows the whole trajectory.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What the scene should look like.",
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
    "set_transform": lambda **kw: sim_rpc("set_transform", **kw),
    "delete_prim": lambda **kw: sim_rpc("delete_prim", **kw),
    "add_reference_to_stage": lambda **kw: sim_rpc("add_reference_to_stage", **kw),
    "write_script": lambda **kw: _exec_write_script(**kw),
    "run_python": lambda **kw: _exec_run_python(**kw),
    "save_stage": lambda **kw: sim_rpc("save_stage", **kw),
    "observe_video": lambda **kw: _exec_observe_video(**kw),
    "observe": lambda **kw: _exec_observe(**kw),
}


# Edit cadence: after this many run_python calls without an intervening
# observe, block further runs. Tighter than the authoring cadence because
# each task-script run is more substantive than a single stage edit.
EDIT_TOOLS = {"run_python"}
EDIT_CADENCE = 3

# Progress tools — what counts as forward motion for the auto-extend logic.
PROGRESS_TOOLS = {
    "write_script",
    "run_python",
    "save_stage",
    "set_transform",
    "delete_prim",
    "add_reference_to_stage",
}
