#!/usr/bin/env python
"""
chat_agent.py — Interactive multi-turn chat with the dt-agent.

Maintains a single conversation history across all turns so the agent
remembers what it built and can answer follow-up questions or modify the
scene incrementally.

Usage:
    python scripts/chat_agent.py [--max-iter N]

Commands at the prompt:
    exit / quit / Ctrl-C / Ctrl-D  — end session
    clear                          — reset conversation history
                                     (the Isaac Sim stage is NOT cleared)
    /save <path>                   — ask the agent to save the current stage
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure the installed package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI
from dt_agent.agent import (
    LLM_BASE_URL,
    LLM_MODEL,
    SIM_BASE_URL,
    _TRACE_DIR,
    _make_tracer,
    run_turn,
)


def _banner():
    print("dt-agent chat  (type 'exit' or Ctrl-D to quit, 'clear' to reset history)")
    print(f"  LLM  : {LLM_MODEL}")
    print(f"  sim  : {SIM_BASE_URL}")
    print()


def _make_slash_save_message(path: str) -> str:
    if not path.strip():
        path = "/workspace/dt-agent/output/chat_scene.usda"
    return f"Save the current stage to {path}"


def main():
    parser = argparse.ArgumentParser(description="Interactive dt-agent chat")
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

    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)

    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = _TRACE_DIR / f"chat_{int(time.time() * 1000)}.jsonl"
    trace = _make_tracer(trace_path)

    _banner()
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
            print("[history cleared — stage in Isaac Sim is unchanged]\n")
            continue

        if raw.lower().startswith("/save"):
            path = raw[len("/save"):].strip()
            raw = _make_slash_save_message(path)

        trace("user_turn", message=raw, history_len=len(history))
        print()  # blank line before agent output

        response_text, hit_cap = run_turn(
            raw,
            history,
            client,
            trace,
            max_iterations=args.max_iter,
            log=True,
            enforce_save=False,
        )

        print(f"\nagent> {response_text}\n")
        if hit_cap:
            print(
                f"  [warn] hit the {args.max_iter}-iteration cap; "
                "try a narrower request or increase --max-iter\n"
            )


if __name__ == "__main__":
    main()
