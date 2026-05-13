#!/usr/bin/env python
"""
chat_tasks.py — Interactive chat with the robot-task agent.

Loads a pre-authored USD scene into Kit, then enters a chat loop where the
agent writes Python task scripts (driving the robot through a task), runs
them in the sim, and observes the result.

Usage:
    python scripts/chat_tasks.py <scene.usda> [--max-iter N]

Example:
    python scripts/chat_tasks.py output/wetlab.usda

Commands at the prompt:
    exit / quit / Ctrl-D — end session
    clear                — reset conversation history (the loaded stage stays)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the installed package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI
from dt_agent.agent import LLM_BASE_URL, LLM_MODEL, SIM_BASE_URL, _TRACE_DIR
from dt_agent.loop import make_tracer, run_turn as _run_turn
from dt_agent.prompts import SYSTEM_PROMPT_TASKS
from dt_agent.tools import sim_rpc
from dt_agent.tools_tasks import (
    EDIT_CADENCE,
    EDIT_TOOLS,
    PROGRESS_TOOLS,
    TOOL_DEFINITIONS,
    TOOL_EXECUTORS,
)


def _container_path_for_scene(host_path: str) -> str:
    """Translate a host-side scene path under <repo>/output/ to the matching
    container-side path so load_stage can read it. Other host paths pass
    through unchanged."""
    p = Path(host_path).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "output"
    try:
        rel = p.relative_to(output_dir)
        return f"/workspace/dt-agent/output/{rel}"
    except ValueError:
        return str(p)


def _banner(scene_path: str, container_scene_path: str):
    print("dt-agent tasks  (type 'exit' or Ctrl-D to quit, 'clear' to reset history)")
    print(f"  scene: {scene_path}")
    print(f"         -> {container_scene_path} (in sim container)")
    print(f"  LLM  : {LLM_MODEL}")
    print(f"  sim  : {SIM_BASE_URL}")
    print()


def run_tasks_turn(message, history, client, trace, max_iter):
    """Wire loop.run_turn with the tasks bundle."""
    return _run_turn(
        message, history, client, trace,
        model=LLM_MODEL,
        system_prompt=SYSTEM_PROMPT_TASKS,
        tool_definitions=TOOL_DEFINITIONS,
        tool_executors=TOOL_EXECUTORS,
        edit_tools=EDIT_TOOLS,
        progress_tools=PROGRESS_TOOLS,
        edit_cadence=EDIT_CADENCE,
        max_iterations=max_iter,
        log=True,
        # For tasks we always require observe before declaring done.
        # save is optional — the agent decides whether to persist the
        # post-task scene.
        enforce_observe=True,
        enforce_save=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive robot-task chat")
    parser.add_argument(
        "scene",
        help="Path to a USD scene file (e.g. output/wetlab.usda). Loaded into "
             "Kit before the chat starts.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="Max tool-call iterations per turn (default 30)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("NV_API_KEY")
    if not api_key:
        print("ERROR: NV_API_KEY not set — check your .env", file=sys.stderr)
        sys.exit(1)

    container_scene_path = _container_path_for_scene(args.scene)

    # Load the scene before opening the chat.
    print(f"[loading scene] {container_scene_path}", file=sys.stderr)
    try:
        result = sim_rpc("load_stage", file_path=container_scene_path)
    except Exception as e:
        print(f"ERROR: load_stage failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not result.get("ok"):
        print(f"ERROR: load_stage refused: {result.get('error')}", file=sys.stderr)
        sys.exit(1)
    print(f"[loaded] prim_count={result['prim_count']}", file=sys.stderr)

    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)

    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = _TRACE_DIR / f"chat_tasks_{int(time.time() * 1000)}.jsonl"
    trace = make_tracer(trace_path)
    trace("scene_loaded", scene=container_scene_path, prim_count=result["prim_count"])

    _banner(args.scene, container_scene_path)
    print(f"  trace: {trace_path}", end="\n\n")

    history: list[dict] = []

    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            print("[bye]")
            break
        if raw.lower() == "clear":
            history.clear()
            print("[history cleared — loaded stage unchanged]\n")
            continue

        trace("user_turn", message=raw, history_len=len(history))
        print()

        response_text, hit_cap = run_tasks_turn(
            raw, history, client, trace, args.max_iter,
        )

        print(f"\nagent> {response_text}\n")
        if hit_cap:
            print(
                f"  [warn] hit the {args.max_iter}-iteration cap; "
                "try a narrower request or increase --max-iter\n"
            )


if __name__ == "__main__":
    main()
