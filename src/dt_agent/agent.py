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
from dt_agent.prompts import SYSTEM_PROMPT_AUTHORING as SYSTEM_PROMPT
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
