"""
loop.py — Generic agent loop machinery.

Drives an LLM tool-calling while-loop with configurable hallucination
guards. The specific tool surface, system prompt, model, and which guards
are active come from the caller — see agent.py for the scene-authoring
assembly, and (later) a robot-tasks assembly that reuses this same loop.

Exports:
- run_turn      — the driver
- make_tracer   — JSONL trace writer
- AUTO_EXTEND_* — auto-extend tuning constants
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


# When the soft iteration cap is reached, auto-extend by this many iterations
# if recent activity shows real progress. Hard ceiling = AUTO_EXTEND_FACTOR ×
# the original soft cap.
AUTO_EXTEND_STEP = 20
AUTO_EXTEND_FACTOR = 3
AUTO_EXTEND_WINDOW = 5  # look-back of iterations to check for progress


def make_tracer(trace_path: Path):
    """Return a tracer callable that appends JSONL records to trace_path."""
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


def _execute_tool(name: str, args: dict, executors: dict) -> str:
    """Execute a tool from `executors` and JSON-encode the result. Caught
    exceptions are surfaced as {"error": ...} so the model can self-correct."""
    if name not in executors:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = executors[name](**args)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    return json.dumps(result, default=str)


def run_turn(
    message: str,
    history: "list[dict[str, Any]]",
    client: "OpenAI",
    trace,
    *,
    model: str,
    system_prompt: str,
    tool_definitions: list,
    tool_executors: dict,
    edit_tools: set,
    progress_tools: set,
    edit_cadence: int,
    observe_tool_name: str = "observe",
    save_tool_name: str = "save_stage",
    max_iterations: int = 30,
    log: bool = True,
    enforce_save: bool = False,
    enforce_observe: bool = False,
) -> tuple[str, bool]:
    """Execute one conversational turn against an existing history.

    Appends `message` as a user turn, runs the tool loop until the model
    emits text with no tool calls, and returns (response_text, hit_cap).

    `history` is mutated in-place — the caller owns it across turns.

    Guard parameters:
    - enforce_observe: intercept finish before observe_tool_name is called.
    - enforce_save:    intercept finish before save_tool_name succeeds.
    - edit_cadence:    block edit_tools after this many edits without an
                       intervening observe.

    The observe/save tool names are parameterized so a non-authoring bundle
    (e.g. robot-tasks) can plug in equivalents without forking the loop.
    """
    history.append({"role": "user", "content": message})
    save_succeeded = False
    stall_count = 0  # consecutive empty-response count; reset on any tool call
    last_vlm_issues: list[str] = []     # issues from the most recent observe call
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
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + history,
                tools=tool_definitions,
                parallel_tool_calls=True,
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
            # Gate 1: require at least one observe before finishing.
            if enforce_observe and last_vlm_satisfied is None:
                nudge = (
                    f"You must call {observe_tool_name}() to verify the result "
                    "before declaring done. Call it now with an appropriate intent."
                )
                history.append({"role": "user", "content": nudge})
                if log:
                    print(f"[agent]  NOTE: intercepted premature finish — {observe_tool_name}() not yet called", file=sys.stderr)
                trace("observe_nudge", iteration=iteration)
                continue

            # Gate 2: require save to succeed before finishing.
            if enforce_save and not save_succeeded:
                nudge = (
                    f"You declared the task done but {save_tool_name} was never "
                    "successfully called — the output has not been written. "
                    "Call it now."
                )
                history.append({"role": "user", "content": nudge})
                if log:
                    print(f"[agent]  NOTE: intercepted premature finish — {save_tool_name} not yet called", file=sys.stderr)
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

            # Guard: enforce observe cadence — block edits after too many
            # have piled up since the last observe.
            if name in edit_tools and edits_since_observe >= edit_cadence:
                blocked_output = json.dumps({
                    "error": (
                        f"edit cadence limit: {edits_since_observe} edits since "
                        f"the last {observe_tool_name}(). Call {observe_tool_name}(...) "
                        "now to verify, then resume."
                    )
                })
                if log:
                    print(
                        f"[agent]  BLOCKED {name} — {edits_since_observe} edits without {observe_tool_name}",
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

            # Guard: block save if the last observe was not satisfied.
            if name == save_tool_name and last_vlm_satisfied is False:
                issues_str = "; ".join(last_vlm_issues) if last_vlm_issues else "see correction_hint"
                blocked_output = json.dumps({
                    "error": (
                        f"{save_tool_name} blocked: the last {observe_tool_name}() "
                        f"returned intent_satisfied=false (issues: {issues_str}). "
                        f"Fix the listed issues and call {observe_tool_name}() again "
                        "to confirm before saving."
                    )
                })
                if log:
                    print(
                        f"[agent]  BLOCKED {save_tool_name} — last {observe_tool_name}() not satisfied "
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
            output = _execute_tool(name, args, tool_executors)

            # Track observe outcomes so the save guard stays current.
            if name == observe_tool_name:
                try:
                    obs = json.loads(output)
                    last_vlm_satisfied = bool(obs.get("intent_satisfied"))
                    last_vlm_issues = obs.get("issues") or []
                    edits_since_observe = 0
                except Exception:
                    pass
            elif name in edit_tools:
                edits_since_observe += 1

            if name == save_tool_name:
                try:
                    if json.loads(output).get("ok"):
                        save_succeeded = True
                        last_vlm_satisfied = None  # reset guard after a successful save
                        edits_since_observe = 0
                except Exception:
                    pass
            if name in progress_tools:
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
        if iteration == soft_cap and soft_cap < hard_cap and not save_succeeded:
            if any(progress_log):
                new_cap = min(soft_cap + AUTO_EXTEND_STEP, hard_cap)
                if log:
                    print(
                        f"[agent]  AUTO-EXTEND iteration cap {soft_cap} -> {new_cap} "
                        f"(still making progress, no {save_tool_name} yet)",
                        file=sys.stderr,
                    )
                trace("auto_extend", from_cap=soft_cap, to_cap=new_cap)
                soft_cap = new_cap

    if log:
        print(f"\n[agent] hit iteration cap ({soft_cap}); stopping.", file=sys.stderr)
    trace("max_iter_reached", soft_cap=soft_cap, hard_cap=hard_cap)
    return f"(hit iteration cap of {soft_cap})", True
