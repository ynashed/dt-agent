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

**Isaac Sim container + MCP pipe:**

```bash
docker compose build      # one-time; ~minutes
docker compose up         # boots Kit headless, starts FastMCP-SSE on :8765
# In another shell:
python scripts/sim_client_smoke.py
```

Expected: the smoke client lists `['get_stage_info']` and prints stage info
returned from a real `omni.usd` call inside the running Isaac Sim instance.

## Layout

```
dt-agent/
├── Dockerfile.isaacsim          # nvcr.io/nvidia/isaac-sim:5.1.0 + fastmcp
├── docker-compose.yml           # GPU passthrough, port 8765, cache volumes
├── hello_inference.py           # Phase 0 LLM-proxy validator
├── pyproject.toml
├── src/dt_agent/
│   ├── __init__.py
│   └── sim_server.py            # Runs in container: Kit + FastMCP-SSE (threaded)
└── scripts/
    └── sim_client_smoke.py      # Runs on host: validates the MCP pipe
```

## Status

Phase 0 — bootstrap. LLM proxy validated. Isaac Sim container + MCP pipe wired
but not yet smoke-tested end-to-end on this machine.
