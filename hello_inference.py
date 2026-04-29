"""
hello_inference.py — Validate the NV inference proxy auth + Responses API wire format.

Reads NV_API_KEY from .env, sends a one-shot prompt to GPT-5.3-codex via the NV
inference proxy, and prints the response. If this works end-to-end, the LLM
plumbing is ready for the next phase.

    python hello_inference.py
"""
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

NV_INFERENCE_BASE_URL = "https://inference-api.nvidia.com/v1"
MODEL = "openai/openai/gpt-5.3-codex"


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("NV_API_KEY")
    if not api_key:
        print(
            "ERROR: NV_API_KEY not set. Copy .env.example to .env and fill in your key.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(api_key=api_key, base_url=NV_INFERENCE_BASE_URL)
    response = client.responses.create(
        model=MODEL,
        input="In one sentence, what is NVIDIA Isaac Sim?",
    )
    print(response.output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
