"""
sim_client_smoke.py — Validate the HTTP RPC pipe to the Isaac Sim container.

Runs on the host. Lists available tools and exercises the core Phase 1
authoring flow end-to-end:

    create_primitive  ->  set_transform  ->  query_stage  ->  save_stage

Plus a `search_assets` probe to confirm the asset roots are reachable.
Uses stdlib urllib so no host-side deps are needed.

    python scripts/sim_client_smoke.py
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8765"
TEST_PRIM = "/World/dt_agent_smoke_sphere"
SAVE_PATH = "/workspace/dt-agent/output/smoke_test.usda"


def _rpc(tool: str, **args):
    req = urllib.request.Request(
        f"{BASE}/rpc",
        data=json.dumps({"tool": tool, "args": args}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read())
    if "error" in body:
        raise RuntimeError(f"{tool} failed: {body['error']}")
    return body["result"]


def main() -> int:
    try:
        # 1. Tool listing
        with urllib.request.urlopen(f"{BASE}/tools", timeout=10) as r:
            tools = json.loads(r.read())
        print(f"[smoke] tools: {tools['tools']}")

        # 2. Stage sanity
        info = _rpc("get_stage_info")
        print(f"[smoke] get_stage_info: {info}")

        # 3. Create a sphere
        created = _rpc("create_primitive", prim_path=TEST_PRIM, prim_type="Sphere")
        print(f"[smoke] create_primitive: {created}")

        # 4. Move it somewhere visible
        moved = _rpc(
            "set_transform",
            prim_path=TEST_PRIM,
            translate=[1.0, 2.0, 0.5],
            scale=[0.25, 0.25, 0.25],
        )
        print(f"[smoke] set_transform: {moved}")

        # 5. Query just our subtree to confirm
        sub = _rpc("query_stage", prim_path=TEST_PRIM, depth=0)
        print(f"[smoke] query_stage({TEST_PRIM}): {sub}")

        # 6. Save to a path inside the container
        saved = _rpc("save_stage", file_path=SAVE_PATH)
        print(f"[smoke] save_stage: {saved}")

        # 7. Asset search probe
        assets = _rpc("search_assets", query="franka", limit=5)
        print(
            f"[smoke] search_assets(query='franka'): "
            f"{len(assets['matches'])} matches "
            f"(truncated={assets['truncated']})"
        )
        for m in assets["matches"][:5]:
            print(f"          {m}")

        return 0

    except urllib.error.URLError as e:
        print(f"[smoke] FAILED to reach {BASE}: {e}", file=sys.stderr)
        print("[smoke] Is the container running? Try: docker compose up", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
