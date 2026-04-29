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
| Sim runtime | `nvcr.io/nvidia/isaac-sim:5.1.0`, exposed via FastMCP over SSE *(planned)* |
| Framework | NeMo Agent Toolkit (NAT) *(planned)* |
| Deployment target | Astra *(post-PoC)* |

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

cp .env.example .env
# Edit .env and set NV_API_KEY (get a key at https://build.nvidia.com)
```

## Validate

**LLM proxy (host-side):**

```bash
python hello_inference.py
```

Expected: a one-sentence description of Isaac Sim, returned via the NV
inference proxy / Responses API.

**Isaac Sim container + RPC pipe:**

```bash
docker compose build      # one-time; ~minutes for first base-image pull
docker compose up         # boots Kit headless, starts HTTP RPC on :8765
# In another shell:
python scripts/sim_client_smoke.py
```

Expected: the smoke client lists the tool surface, then walks
`create_primitive` → `set_transform` → `query_stage` → `save_stage` against
a sphere at `/World/dt_agent_smoke_sphere`, finishing with a probe of
`search_assets` against the in-image asset roots. Each step prints the
`{result: ...}` returned by Kit.

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
| `search_assets` | `query, limit=30, roots?` | `{matches, truncated}` |

## Layout

```
dt-agent/
├── Dockerfile.isaacsim          # nvcr.io/nvidia/isaac-sim:5.1.0 (vanilla; root user)
├── docker-compose.yml           # GPU passthrough, port 8765, cache volumes
├── hello_inference.py           # Phase 0 LLM-proxy validator
├── pyproject.toml
├── src/dt_agent/
│   ├── __init__.py
│   └── sim_server.py            # Runs in container: Kit + stdlib HTTP RPC (threaded)
└── scripts/
    └── sim_client_smoke.py      # Runs on host: validates the RPC pipe
```

## Status

**Phase 0 — bootstrap (done):** LLM proxy validated; Isaac Sim container +
HTTP RPC pipe smoke-tested end-to-end.

**Phase 1 — open-loop authoring (in progress):** tool surface for stage
inspection and edit landed (`query_stage`, `create_primitive`,
`add_reference_to_stage`, `set_transform`, `save_stage`, `search_assets`).
Next: scripted demo flow exercising the tools to compose a workcell scene
without an LLM in the loop, then layer the agent in Phase 2.
