"""
observe_capture.py — Send a captured viewport PNG to Cosmos Reason and
print the structured Observation.

Standalone CLI for testing the VLM step in isolation, before plugging it
into the agent loop. Useful for:
- Verifying the API is reachable and the auth works
- Iterating on the system prompt to get well-shaped JSON back
- Sanity-checking that the model's spatial reasoning matches yours

    python scripts/observe_capture.py <image.png> "<intent string>"

Example:
    python scripts/observe_capture.py output/captures/capture_<...>.png \\
        "the UR10e arm should sit on the table; three microplates should \\
         be stacked on the conveyor to its right"
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make `src/` importable when running as a one-off script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dt_agent.vlm import observe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Path to a PNG/JPEG capture to observe.")
    parser.add_argument(
        "intent",
        help="What the captured scene should look like, in plain English.",
    )
    args = parser.parse_args()

    load_dotenv()

    try:
        obs = observe(args.image, args.intent)
    except FileNotFoundError:
        print(f"[observe] image not found: {args.image}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[observe] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(obs.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
