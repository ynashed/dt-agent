"""
sim_client_smoke.py — Validate the HTTP RPC pipe to the Isaac Sim container.

Runs on the host. Lists available tools and exercises the core Phase 1
authoring flow end-to-end:

    create_primitive  ->  set_transform  ->  query_stage  ->  save_stage

Plus a `search_assets` probe to confirm the asset roots are reachable.
Uses stdlib urllib so no host-side deps are needed.

Usage:
    python scripts/sim_client_smoke.py                  # standard smoke flow
    python scripts/sim_client_smoke.py --probe-s3       # + check Kit can
                                                          fetch USD over HTTPS
    python scripts/sim_client_smoke.py --probe-s3 \\
        --s3-url <full URL>                             # try a different URL
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8765"
TEST_PRIM = "/World/dt_agent_smoke_sphere"
SAVE_PATH = "/workspace/dt-agent/output/smoke_test.usda"

# Probe target — a known-named asset on NVIDIA's production OpenUSD CDN.
# If this 404s the URL pattern may have shifted (use --s3-url to override).
DEFAULT_S3_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
)
S3_PROBE_PRIM = "/World/dt_agent_s3_probe"


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

        # 4. Move it somewhere visible to the default observation camera
        # (centered around the workcell origin at ~(0.6, 0, 0.45)).
        moved = _rpc(
            "set_transform",
            prim_path=TEST_PRIM,
            translate=[0.0, 0.5, 0.5],
            scale=[0.25, 0.25, 0.25],
        )
        print(f"[smoke] set_transform: {moved}")

        # 5. Query just our subtree to confirm
        sub = _rpc("query_stage", prim_path=TEST_PRIM, depth=0)
        print(f"[smoke] query_stage({TEST_PRIM}): {sub}")

        # 6. Save to a path inside the container
        saved = _rpc("save_stage", file_path=SAVE_PATH)
        print(f"[smoke] save_stage: {saved}")

        # 7. Asset search probe — exercises both catalog and filesystem sources
        assets = _rpc("search_assets", query="ur10", limit=10)
        print(
            f"[smoke] search_assets(query='ur10'): "
            f"{len(assets['matches'])} matches "
            f"(truncated={assets['truncated']})"
        )
        for m in assets["matches"][:5]:
            src = m.get("source", "?")
            label = m.get("name") or m["path"]
            verified = ""
            if src == "catalog":
                verified = " [verified]" if m.get("verified") else " [unverified]"
            print(f"          [{src}]{verified} {label}")
            if m.get("name") and m.get("path") != m.get("name"):
                print(f"             {m['path']}")

        # 8. Viewport capture — fixed-pose observation camera, PNG to ./output/captures/
        cap = _rpc("capture_viewport")
        print(f"[smoke] capture_viewport: {cap}")

        return 0

    except urllib.error.URLError as e:
        print(f"[smoke] FAILED to reach {BASE}: {e}", file=sys.stderr)
        print("[smoke] Is the container running? Try: docker compose up", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


def s3_probe(url: str) -> int:
    """Probe whether Kit's URL resolver inside the container can fetch a USD
    asset over HTTPS. Adds a reference at S3_PROBE_PRIM, then inspects the
    subtree — real descendants mean the fetch landed."""
    print(f"[s3-probe] target: {url}")
    try:
        added = _rpc("add_reference_to_stage", prim_path=S3_PROBE_PRIM, usd_path=url)
        print(f"[s3-probe] add_reference_to_stage: {added}")

        sub = _rpc("query_stage", prim_path=S3_PROBE_PRIM, depth=2)
        prims = sub.get("prims", [])
        # First entry is the root we created; descendants only count if there
        # are entries past index 0.
        descendants = prims[1:]
        if not descendants:
            print(f"[s3-probe] NO DESCENDANTS — Kit didn't resolve {url}.", file=sys.stderr)
            print(
                "[s3-probe] Either the container has no HTTPS egress, or the "
                "URL is wrong. Try --s3-url with a known-good asset URL.",
                file=sys.stderr,
            )
            return 1
        print(f"[s3-probe] OK — {len(descendants)} descendants under {S3_PROBE_PRIM}:")
        for p in descendants[:8]:
            print(f"            {p['path']}  [{p['type']}]")
        if len(descendants) > 8:
            print(f"            ... ({len(descendants) - 8} more)")
        return 0
    except Exception as e:
        print(f"[s3-probe] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--probe-s3",
        action="store_true",
        help="After the smoke flow, probe whether Kit can fetch USD assets "
             "from NVIDIA's S3 OpenUSD CDN over HTTPS.",
    )
    parser.add_argument(
        "--s3-url",
        default=DEFAULT_S3_URL,
        help="Override the URL used for the S3 probe.",
    )
    args = parser.parse_args()

    rc = main()
    if args.probe_s3 and rc == 0:
        rc = s3_probe(args.s3_url)
    sys.exit(rc)
