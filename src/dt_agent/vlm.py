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
import re
import sys
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

# Default to the local NIM at http://localhost:8000/v1 — matches the
# `vlm` service in docker-compose.yml. Override via NV_VLM_BASE_URL to
# point at the NVIDIA-hosted endpoint (https://integrate.api.nvidia.com/v1)
# or any other OpenAI-compat proxy.
VLM_BASE_URL = os.environ.get("NV_VLM_BASE_URL", "http://localhost:8000/v1")
VLM_MODEL = os.environ.get("NV_VLM_MODEL", "nvidia/cosmos-reason2-8b")


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

Emit ONLY the JSON object. No markdown fences, no preamble, no commentary."""


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


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_INCOMPLETE_THINK = re.compile(r"<think>.*$", flags=re.DOTALL)
_MISSING_COMMA = re.compile(r'("\s*)\n(\s*)(?=")')


def _strip_fences(text: str) -> str:
    """Defensive cleanup of the model's response before json.loads.

    Handles two things despite the system prompt telling the model not to:
      1. ```json ... ``` markdown fences (sometimes added by chat models).
      2. <think>...</think> reasoning blocks (Cosmos Reason can emit
         chain-of-thought before the JSON if reasoning mode is on).
    """
    text = text.strip()
    text = _THINK_BLOCK.sub("", text).strip()
    # Strip an incomplete think block (token limit hit before </think>).
    text = _INCOMPLETE_THINK.sub("", text).strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _try_repair_json(text: str) -> str | None:
    """Minimal fixup for the most common LLM JSON-emission bug: a missing
    comma between two adjacent string-keyed properties (the model emits
    `"value"\\n  "next_key":` instead of `"value",\\n  "next_key":`).
    Returns None if no repair was applied."""
    repaired = _MISSING_COMMA.sub(r'\1,\n\2', text)
    return repaired if repaired != text else None


def _try_repair_truncated_json(text: str) -> str | None:
    """Repair JSON cut off by a token limit — the most common form is an
    unterminated string in the `observed` field. Closes open strings, injects
    any missing required fields with safe defaults, and closes the object.
    Returns None if the result still doesn't parse."""
    candidate = _MISSING_COMMA.sub(r'\1,\n\2', text).rstrip()

    # Count unescaped double-quotes; odd count means we're inside a string.
    if sum(1 for _ in re.finditer(r'(?<!\\)"', candidate)) % 2 == 1:
        candidate += '"'

    # Detect which required fields are already present.
    has_issues = '"issues"' in candidate
    has_hint = '"correction_hint"' in candidate
    is_satisfied = (
        '"intent_satisfied": true' in candidate
        or '"intent_satisfied":true' in candidate
    )

    extras = []
    if not has_issues:
        extras.append('"issues": []')
    if not has_hint:
        hint = 'null' if is_satisfied else '"Scene could not be fully assessed — observation was truncated"'
        extras.append(f'"correction_hint": {hint}')

    if extras:
        if not candidate.rstrip().endswith(','):
            candidate += ','
        candidate += ' ' + ', '.join(extras)

    candidate += '}'

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "://vlm:", "://vlm/")


def _is_local_endpoint(url: str) -> bool:
    return any(needle in url for needle in _LOCAL_HOSTS)


def observe(
    image_path: str | Path,
    intent: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
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
        # vLLM's xgrammar-backed structured output forces the model to
        # emit valid JSON — fixes the common "missing comma between
        # properties" bug we saw without this. Some non-vLLM proxies
        # ignore this param silently; the repair fallback below catches
        # those cases.
        response_format={"type": "json_object"},
    )

    raw = (response.choices[0].message.content or "").strip()
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as primary_err:
        # Try repairs in order: comma fix → truncation fix.
        for label, attempt in [
            ("comma-repair", _try_repair_json(cleaned)),
            ("truncation-repair", _try_repair_truncated_json(cleaned)),
        ]:
            if attempt is None:
                continue
            try:
                data = json.loads(attempt)
                print(f"[vlm] WARN: {label} fixed malformed JSON ({primary_err})", file=sys.stderr)
                break
            except json.JSONDecodeError:
                continue
        else:
            raise RuntimeError(
                f"VLM response was not valid JSON ({primary_err}). "
                f"First 500 chars of response: {cleaned[:500]!r}"
            ) from primary_err
    return Observation.model_validate(data)
