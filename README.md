# dt-agent

Isaac Sim digital twin authoring agent — PoC.

Builds digital twins from a text spec, optional user assets, and the Isaac Sim
shipped asset/material library. The agent loop is **plan → edit (USD/Python) →
execute → observe (VLM) → reflect**.

## Stack

| Layer | Technology |
|---|---|
| Coder/Planner LLM | GPT-5.3-codex via NV inference proxy (Responses API) |
| VLM | Cosmos Reason via build.nvidia.com *(planned)* |
| Sim runtime | `nvcr.io/nvidia/isaac-sim:5.1.0`, exposed via stdlib HTTP RPC on `:8765` |
| Framework | NeMo Agent Toolkit (NAT) *(planned)* |
| Deployment target | Astra *(post-PoC)* |

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

cp .env.example .env
# Edit .env:
#   NV_API_KEY      — for GPT-5.3-codex on inference-api.nvidia.com
#   NV_VLM_API_KEY  — for Cosmos Reason 2 8B on build.nvidia.com (separate
#                     auth surface; generate from the "Get API Key" panel at
#                     https://build.nvidia.com/nvidia/cosmos-reason2-8b)
```

## Validate

**LLM proxy (host-side):**

```bash
python hello_inference.py
```

Expected: a one-sentence description of Isaac Sim, returned via the NV
inference proxy / Responses API.

**Isaac Sim container + Cosmos Reason 2 8B NIM:**

The compose stack runs two services side-by-side: `isaac-sim` (rendering +
RPC) on GPU 0, and `vlm` (Cosmos Reason 2 8B) on GPU 1.

One-time setup:

```bash
# Log in to NGC so docker can pull the NIM image.
# Get an NGC API key at https://ngc.nvidia.com/setup/api-key.
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY

# Pre-pull the NIM (~tens of GB; first run will also fetch model weights).
docker compose pull vlm
```

Run:

```bash
docker compose build      # one-time for isaac-sim; ~minutes for first base-image pull
docker compose up         # starts both services
                          # isaac-sim:  http://localhost:8765
                          # vlm:        http://localhost:8000  (~10 min first warmup)
# In another shell, once both are up:
python scripts/sim_client_smoke.py
```

Expected: the smoke client lists the tool surface, then walks
`create_primitive` → `set_transform` → `query_stage` → `save_stage` against
a sphere at `/World/dt_agent_smoke_sphere`, finishing with a probe of
`search_assets` (catalog + filesystem). Each step prints the
`{result: ...}` returned by Kit.

Add `--probe-s3` to additionally verify Kit's URL resolver can fetch USDs
from NVIDIA's OpenUSD CDN over HTTPS — required if you want to use the
catalog's S3 URLs.

**Scripted workcell demo (no LLM yet):**

```bash
python scripts/build_workcell.py
```

Composes a benchtop workcell — table (Cube primitive), UR10e arm
(USD reference from the CDN), conveyor (Cube primitive), three microplate
stand-ins (Cube primitives) — and saves to
`/workspace/dt-agent/output/workcell.usda` inside the container. Proves the
RPC tool surface composes into a real scene before the LLM is in the loop.

**VLM observation (Phase 2.0):**

```bash
python scripts/observe_capture.py output/captures/capture_<latest>.png \
    "the UR10e should sit on the table; three microplates should be on the conveyor"
```

Sends the PNG to the local Cosmos Reason 2 8B NIM at
`http://localhost:8000` (default in `.env.example`) and prints a structured
`Observation` JSON: `{intent_satisfied, observed, issues, correction_hint}`.
That `correction_hint` is what the agent loop in Phase 2.5 will feed back
to the LLM as the "what to fix next" signal.

To use the NVIDIA-hosted variant on `integrate.api.nvidia.com` instead,
unset `NV_VLM_BASE_URL` and supply `NV_VLM_API_KEY` from the build.nvidia.com
panel (see `.env.example`).

## RPC contract

The container's bridge is a stdlib HTTP server (no external Python deps
inside the Isaac Sim image — that avoids version clashes with Kit's
bundled vendored libraries). The MCP layer NAT will consume lives on the
agent host (added later, in its own clean Python env).

- `GET  /tools`  →  `{"tools": ["get_stage_info", ...]}`
- `POST /rpc`    →  body `{"tool": "<name>", "args": {...}}`
                    →  `{"result": <json>}`  on success
                    →  `{"error": "..."}`  on failure (also non-200)

Phase 1 tool surface:

| Tool | Args | Returns |
|---|---|---|
| `get_stage_info` | — | `{loaded, url, prim_count}` |
| `query_stage` | `prim_path="/", depth=3` | `{root, prims: [{path, type, translate?}], truncated}` |
| `create_primitive` | `prim_path, prim_type="Xform"` | `{ok, prim_path, type}` |
| `add_reference_to_stage` | `usd_path, prim_path` | `{ok, prim_path, usd_path}` |
| `set_transform` | `prim_path, translate?, rotate?, scale?` | `{ok, prim_path}` |
| `save_stage` | `file_path` | `{ok, file_path}` |
| `search_assets` | `query, limit=30, roots?, sources?` | `{matches: [{path, source, name?, description?, category?, verified?}], truncated}` |
| `capture_viewport` | `camera_path?, eye?, target?, resolution?, file_path?` | `{ok, file_path, camera_path, resolution, size}` |

`search_assets` queries two sources by default — the curated catalog at
`catalog/asset_catalog.json` (HTTPS URLs on NVIDIA's OpenUSD CDN) and the
local filesystem under `_DEFAULT_ASSET_ROOTS`. Pass `sources=["catalog"]`
or `sources=["filesystem"]` to restrict. Catalog matches include `name`,
`description`, `category`, and a `verified` flag indicating whether the
URL was confirmed to fetch successfully against this image.

`capture_viewport` renders the scene from a fixed-pose observation camera
(default `/World/_dt_observation_cam`, look-at the workcell origin from
`(3, 3, 2)`) via `omni.replicator.core` and saves a PNG to
`/workspace/dt-agent/output/captures/` (which the host sees under
`./output/captures/`). The default camera is created on first call and its
transform is refreshed each call. Pass an explicit `camera_path` to render
from an existing camera prim with whatever transform it already has.

## Layout

```
dt-agent/
├── Dockerfile.isaacsim          # nvcr.io/nvidia/isaac-sim:5.1.0 (vanilla; root user)
├── docker-compose.yml           # GPU passthrough, port 8765, cache + code mounts
├── hello_inference.py           # Phase 0 LLM-proxy validator
├── pyproject.toml
├── catalog/
│   └── asset_catalog.json       # Curated NVIDIA OpenUSD CDN URLs
├── src/dt_agent/
│   ├── __init__.py
│   ├── sim_server.py            # Runs in container: Kit + stdlib HTTP RPC (threaded)
│   └── vlm.py                   # Runs on host: Cosmos Reason wrapper -> Observation
└── scripts/
    ├── sim_client_smoke.py      # Runs on host: validates the RPC pipe (--probe-s3 optional)
    ├── build_workcell.py        # Runs on host: composes a benchtop workcell scene
    └── observe_capture.py       # Runs on host: VLM observation of a captured PNG
```

## Status

**Phase 0 — bootstrap (done):** LLM proxy validated; Isaac Sim container +
HTTP RPC pipe smoke-tested end-to-end.

**Phase 1 — open-loop authoring (done):** tool surface for stage inspection
and edit (`query_stage`, `create_primitive`, `add_reference_to_stage`,
`set_transform`, `save_stage`, `search_assets`). Curated asset catalog on
top of NVIDIA's OpenUSD CDN with HTTPS fetch confirmed. Scripted workcell
demo composes table + UR10e + conveyor + microplates without an LLM, opens
in Isaac Sim GUI on the host.

**Phase 1.5 — viewport capture (done):** `capture_viewport` RPC renders a
fixed-pose observation camera to PNG. Saves to `./output/captures/`.

**Phase 2.0 — VLM observation (in progress):** `dt_agent.vlm.observe(image, intent)`
sends a captured frame to **Cosmos Reason 2 8B** at
`integrate.api.nvidia.com` and returns a Pydantic-validated `Observation`
(intent_satisfied, observed, issues, correction_hint). Standalone CLI:
`scripts/observe_capture.py`.

**Phase 2.5 — agent loop:** wire GPT-5.3-codex + the RPC tools + the VLM
observer into a plan→edit→capture→observe→reflect loop. NAT graph or
thin custom orchestration — TBD.
