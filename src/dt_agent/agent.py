"""
agent.py — Scene-authoring agent entry point.

Wires the scene-authoring SYSTEM_PROMPT, tool bundle (from tools.py), and
default guard configuration into the generic loop in loop.py. Exposes the
run_turn / run entry points that scripts/chat_agent.py and
scripts/run_agent.py call.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from dt_agent.loop import make_tracer, run_turn as _run_turn
from dt_agent.tools import (
    EDIT_CADENCE,
    EDIT_TOOLS,
    PROGRESS_TOOLS,
    TOOL_DEFINITIONS,
    TOOL_EXECUTORS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.environ.get(
    "NV_LLM_BASE_URL", "https://inference-api.nvidia.com/v1"
)
LLM_MODEL = os.environ.get("NV_LLM_MODEL", "aws/anthropic/bedrock-claude-opus-4-7")
SIM_BASE_URL = os.environ.get("DT_AGENT_SIM_URL", "http://localhost:8765")

# Trace logs for each agent run.
_TRACE_DIR = Path(__file__).resolve().parents[2] / "output" / "agent_traces"


SYSTEM_PROMPT = """You are a digital-twin authoring agent for NVIDIA Isaac Sim.
You receive a natural-language goal describing a desired scene, and you have
tools to inspect, edit, and observe the scene to make it match the goal.

Workflow:
1. Survey: call `query_stage` to see what's already in the scene.
2. Plan: identify the prims you need with their target paths and transforms.
3. Build: use `add_reference_to_stage`, `set_transform`, and `get_prim_bounds`
   to compose the scene. Every object in the scene must be a referenced USD
   asset — geometry primitives (Cube, Sphere, Cylinder, etc.) are not available.
   - Scope: add only what the goal explicitly names. A bare "warehouse" or
     "room" is one named component, not a request for individual ceiling
     tiles, structural beams, or scaffolding — only add those if the goal
     lists them.
   - Asset use: for each component the goal names, call `search_assets_ai`.
     If it returns a relevant result, you MUST use `add_reference_to_stage`
     with one of those URLs. If search returns nothing usable, try alternate
     search terms before giving up — e.g. "workbench" instead of "table".
   - Tiled/modular assets (wall panels, floor tiles): load ONE instance
     under a probe path like `/World/_Probe/<name>`, call `get_prim_bounds`
     to measure, then `delete_prim` the probe so it does NOT appear in
     observe() captures. Then place the real tiles with
     `add_reference_to_stage` + `set_transform`. Do NOT abandon the asset
     and tile with a proxy — only real referenced USDs may appear in the scene.
4. Validate: call `observe(intent)` after each meaningful chunk of edits —
   not at the end of the build. A "chunk" is one logical addition (e.g. all
   walls, the floor + ceiling, the lighting pass). The framework will block
   further edits if you exceed ~8 edit calls without an intervening observe.
   The vision model returns `{intent_satisfied, observed, issues,
   correction_hint}`. If the observation contradicts what `query_stage`
   reports, trust `query_stage` for ground truth — the VLM may misidentify
   visually ambiguous geometry. A completely black or near-black image means
   no light reaches the camera — do NOT save or declare done. For enclosed
   spaces add interior lights with `add_light` before observing again.
5. Iterate: when `intent_satisfied` is false, address the `issues` list one
   item at a time, then re-observe.
6. Save: when satisfied, `save_stage` to the requested file path and reply
   with a brief plain-text summary (no tool calls) to signal completion.

Conventions:
- Z-up world. Units are meters.
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
- Materials: use `search_materials` + `bind_material` to apply surface materials
  to any prim. Call `search_materials` first to get the MDL URL, then
  `bind_material(prim_path, material_url)`. Bind after placing and transforming
  the prim, before the next `observe` call.
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
    """Scene-authoring turn. Thin wrapper over loop.run_turn that fills in
    the authoring system prompt, tool bundle, and guard config."""
    return _run_turn(
        message, history, client, trace,
        model=LLM_MODEL,
        system_prompt=SYSTEM_PROMPT,
        tool_definitions=TOOL_DEFINITIONS,
        tool_executors=TOOL_EXECUTORS,
        edit_tools=EDIT_TOOLS,
        progress_tools=PROGRESS_TOOLS,
        edit_cadence=EDIT_CADENCE,
        max_iterations=max_iterations,
        log=log,
        enforce_save=enforce_save,
        enforce_observe=enforce_observe,
    )


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
    trace = make_tracer(trace_path)

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
