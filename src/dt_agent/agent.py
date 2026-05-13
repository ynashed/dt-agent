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
from pathlib import Path
from typing import Any

from openai import OpenAI

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


# When the soft iteration cap is reached, auto-extend by this many iterations
# if recent activity shows real progress. Hard ceiling = AUTO_EXTEND_FACTOR ×
# the original soft cap.
AUTO_EXTEND_STEP = 20
AUTO_EXTEND_FACTOR = 3
AUTO_EXTEND_WINDOW = 5  # look-back of iterations to check for progress


def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return a JSON-encoded string (what the Responses
    API expects as function_call_output.output). Exceptions are caught and
    surfaced to the model so it can self-correct."""
    if name not in TOOL_EXECUTORS:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = TOOL_EXECUTORS[name](**args)
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
    edits_since_observe = 0  # cadence counter; reset on observe()
    soft_cap = max_iterations
    hard_cap = max_iterations * AUTO_EXTEND_FACTOR
    progress_log: list[bool] = []  # one entry per iteration; True if any progress tool succeeded

    iteration = 0
    while iteration < soft_cap:
        iteration += 1
        if log:
            print(f"\n[agent] -- iteration {iteration}/{soft_cap} --", file=sys.stderr)

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
        iter_made_progress = False  # set True when any progress tool succeeds this iteration
        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments) if call.function.arguments else {}

            # Guard: enforce observe cadence — block further edits when too many
            # have piled up since the last observe().
            if name in EDIT_TOOLS and edits_since_observe >= EDIT_CADENCE:
                blocked_output = json.dumps({
                    "error": (
                        f"edit cadence limit: {edits_since_observe} edits since "
                        f"the last observe(). Call observe(intent=...) now to "
                        "verify the scene before any further edits, then resume."
                    )
                })
                if log:
                    print(
                        f"[agent]  BLOCKED {name} — {edits_since_observe} edits without observe",
                        file=sys.stderr,
                    )
                trace(
                    "cadence_blocked",
                    iteration=iteration,
                    name=name,
                    edits_since_observe=edits_since_observe,
                )
                history.append({"role": "tool", "tool_call_id": call.id, "content": blocked_output})
                continue

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
                    edits_since_observe = 0
                except Exception:
                    pass
            elif name in EDIT_TOOLS:
                edits_since_observe += 1

            if name == "save_stage":
                try:
                    if json.loads(output).get("ok"):
                        save_stage_succeeded = True
                        last_vlm_satisfied = None  # reset guard after a successful save
                        edits_since_observe = 0
                except Exception:
                    pass
            if name in PROGRESS_TOOLS:
                try:
                    if json.loads(output).get("ok"):
                        iter_made_progress = True
                except Exception:
                    pass
            if log:
                trim = output if len(output) <= 240 else output[:240] + "..."
                print(f"[agent]      = {trim}", file=sys.stderr)
            trace("tool_call", iteration=iteration, name=name, args=args, output=output)
            history.append({"role": "tool", "tool_call_id": call.id, "content": output})

        progress_log.append(iter_made_progress)
        if len(progress_log) > AUTO_EXTEND_WINDOW:
            progress_log.pop(0)

        # Auto-extend the soft cap when we hit it but the agent is still
        # actively making productive edits and hasn't saved yet. Bounded by
        # hard_cap so a runaway loop still terminates.
        if iteration == soft_cap and soft_cap < hard_cap and not save_stage_succeeded:
            if any(progress_log):
                new_cap = min(soft_cap + AUTO_EXTEND_STEP, hard_cap)
                if log:
                    print(
                        f"[agent]  AUTO-EXTEND iteration cap {soft_cap} -> {new_cap} "
                        f"(still making progress, no save_stage yet)",
                        file=sys.stderr,
                    )
                trace("auto_extend", from_cap=soft_cap, to_cap=new_cap)
                soft_cap = new_cap

    if log:
        print(f"\n[agent] hit iteration cap ({soft_cap}); stopping.", file=sys.stderr)
    trace("max_iter_reached", soft_cap=soft_cap, hard_cap=hard_cap)
    return f"(hit iteration cap of {soft_cap})", True


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
