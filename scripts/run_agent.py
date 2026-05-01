"""
run_agent.py — Drive the dt-agent loop from the command line.

Usage:
    python scripts/run_agent.py "<goal in plain English>"

Example:
    python scripts/run_agent.py \\
        "Build a benchtop workcell at the origin: a 1m square table with a \\
         UR10e arm on top, a conveyor belt 1.2m to its right, and three \\
         microplates stacked on the conveyor. Save to \\
         /workspace/dt-agent/output/agent_workcell.usda."
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make `src/` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dt_agent.agent import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("goal", help="Natural-language description of the scene to build.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="Hard cap on agent iterations. Default 30.",
    )
    args = parser.parse_args()

    load_dotenv()
    return run(args.goal, max_iterations=args.max_iter)


if __name__ == "__main__":
    sys.exit(main())
