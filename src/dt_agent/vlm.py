"""
vlm.py — VLM observation wrapper for the dt-agent loop.

Wraps NVIDIA's Cosmos Reason 2 8B (hosted at integrate.api.nvidia.com,
the build.nvidia.com NIM proxy) with a Pydantic-schema-constrained
`observe(image_path, intent) -> Observation` function. Cosmos Reason is
specialized for physical-world reasoning, which is what we want for
"did this edit have the intended visible effect on the scene?"

Lives on the agent host — Kit's Python doesn't see this module. The
container only needs to know how to render a PNG; everything VLM-related
runs in the host venv with normal modern deps.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

# Default to the local NIM at http://localhost:8000/v1 — matches the
# `vlm` service in docker-compose.yml. Override via NV_VLM_BASE_URL to
# point at the NVIDIA-hosted endpoint (https://integrate.api.nvidia.com/v1)
# or any other OpenAI-compat proxy.
VLM_BASE_URL = os.environ.get("NV_VLM_BASE_URL", "http://localhost:8000/v1")
VLM_MODEL = os.environ.get("NV_VLM_MODEL", "nvidia/cosmos-reason2-8b")

# Hard caps enforced at decode time by vLLM's xgrammar JSON-schema backend.
# These are stricter than the prompt rule (which Cosmos Reason 2 8B has
# been ignoring) and prevent the multi-KB `observed` strings that have
# repeatedly blown past max_tokens and broken our repair fallbacks.
_OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent_satisfied": {"type": "boolean"},
        "observed": {"type": "string", "maxLength": 240},
        "issues": {
            "type": "array",
            "items": {"type": "string", "maxLength": 120},
        },
        "correction_hint": {"type": ["string", "null"]},
    },
    "required": ["intent_satisfied", "observed", "issues", "correction_hint"],
    "additionalProperties": False,
}


class Observation(BaseModel):
    """Structured response the VLM emits about a captured frame."""

    intent_satisfied: bool = Field(
        description="True if the captured frame visibly matches the stated intent."
    )
    observed: str = Field(
        description="One or two sentence description of what is actually in the frame."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Discrete things that look wrong or unexpected. Empty if intent_satisfied is true.",
    )
    correction_hint: str | None = Field(
        default=None,
        description="Natural-language guidance for the LLM about what to change next. Null if no fix needed.",
    )


SYSTEM_PROMPT = """You are a vision-language observer for a digital-twin authoring agent.
You receive a rendered frame of an Isaac Sim scene plus an `intent` string
describing what the scene SHOULD contain. Compare what you see to the intent.

Recognition guidance:
- Robotic arms (UR10e, Franka, KUKA, etc.) appear as articulated metallic
  structures with multiple revolute joints linking arm segments.
- Tables, conveyors, shelves, workbenches appear as rectangular slabs.
- Boxes, microplates, bins appear as small cuboid objects.
Match what you see against the component NAMES in the intent — don't insist
on photo-realistic appearance. A "metallic structure with joints" on a slab
is the named robot arm.

Reply with ONLY a JSON object with these keys:

{
  "intent_satisfied": <bool>,
  "observed": <string, 1-2 sentences describing what is actually in the frame>,
  "issues": [<string>, ...],
  "correction_hint": <string or null>
}

Rules — follow these EXACTLY:
- intent_satisfied is true ONLY if every component named in the intent is
  visibly present and roughly correctly placed.
- If intent_satisfied is false: issues MUST contain at least one concrete
  problem AND correction_hint MUST be a non-null actionable string.
- If intent_satisfied is true: issues MUST be [] and correction_hint MUST
  be null.
- Be terse: name what you see, do not narrate lighting, mood, or style."""


def _data_uri(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    if suffix not in {"png", "jpeg"}:
        raise ValueError(f"unsupported image type: .{suffix}")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{b64}"


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "://vlm:", "://vlm/")


def _is_local_endpoint(url: str) -> bool:
    return any(needle in url for needle in _LOCAL_HOSTS)


_FAILED_DIR = Path(__file__).resolve().parents[2] / "output" / "vlm_failures"


def _dump_failed_response(raw: str, intent: str, err: Exception) -> str:
    """Write a malformed VLM response to disk for offline inspection.
    Includes the intent (for reproduction) and the parse error alongside
    the raw text. Should be rare under json_schema enforcement, but kept
    as a safety net for proxies that silently ignore response_format."""
    try:
        _FAILED_DIR.mkdir(parents=True, exist_ok=True)
        import time as _time
        path = _FAILED_DIR / f"failed_{int(_time.time() * 1000)}.txt"
        path.write_text(
            f"intent: {intent}\n"
            f"error: {type(err).__name__}: {err}\n"
            f"--- raw response ({len(raw)} chars) ---\n"
            f"{raw}\n"
        )
        return str(path)
    except Exception as e:
        return f"(failed to dump: {type(e).__name__}: {e})"


def observe(
    image_path: str | Path,
    intent: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
) -> Observation:
    """Send a captured frame to Cosmos Reason and return a parsed Observation.

    Raises FileNotFoundError if image_path doesn't exist, RuntimeError on
    API/parse failures, and pydantic.ValidationError if the model returns
    valid JSON but with the wrong shape.
    """
    effective_url = base_url or VLM_BASE_URL
    api_key = api_key or os.environ.get("NV_VLM_API_KEY") or os.environ.get("NV_API_KEY")
    if not api_key:
        if _is_local_endpoint(effective_url):
            # Self-hosted NIM (e.g. the docker-compose vlm service at
            # http://localhost:8000/v1) doesn't authenticate the chat
            # completions endpoint — but the OpenAI SDK requires a non-empty
            # api_key parameter, so pass a placeholder.
            api_key = "not-needed"
        else:
            raise RuntimeError(
                "No API key available — set NV_API_KEY (or NV_VLM_API_KEY) "
                "in the environment or .env, or point NV_VLM_BASE_URL at a "
                "local NIM (e.g. http://localhost:8000/v1)."
            )

    client = OpenAI(api_key=api_key, base_url=effective_url)

    response = client.chat.completions.create(
        model=model or VLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Intent: {intent}"},
                    {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        # json_schema (vLLM xgrammar) enforces the shape AND maxLength caps
        # at decode time, so the model cannot emit a 19 KB `observed` field
        # that overruns max_tokens mid-string. Strictly stronger than
        # json_object, which only enforces "valid JSON".
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "observation",
                "strict": True,
                "schema": _OBSERVATION_SCHEMA,
            },
        },
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        dump_path = _dump_failed_response(raw, intent, err)
        raise RuntimeError(
            f"VLM response was not valid JSON ({err}). "
            f"Full response written to {dump_path} for inspection."
        ) from err
    return Observation.model_validate(data)
