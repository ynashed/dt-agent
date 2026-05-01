"""
build_workcell.py — Compose a benchtop workcell using the Phase 1 RPC tools.

No LLM in the loop yet. This proves the tool surface composes into a
coherent scene before we let GPT-5.3-codex drive the same calls.

Result on /workspace/dt-agent/output/workcell.usda:

  /World/workcell/
    table         Cube  ~1m x 1m x 0.04m   at origin, top at z=0.42m
    ur10e         Xform with USD reference to UR10e from NVIDIA's CDN,
                  mounted on the table top
    conveyor      Cube  ~1.2m x 0.4m x 0.05m   1.2m to the table's right
    microplate_0  Cube  ~0.12m x 0.08m x 0.015m   stacked on the conveyor
    microplate_1  Cube  ditto, 1 plate-thickness above
    microplate_2  Cube  ditto, 2 plate-thicknesses above

Note on USD Cube semantics: the default Cube prim has an authored size of
2.0 (spans -1..+1 along each axis = 2m total). The `scale` we set acts as
half-extents in meters — e.g. scale=[0.5, 0.5, 0.02] gives 1m x 1m x 0.04m.

    python scripts/build_workcell.py
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8765"
SAVE_PATH = "/workspace/dt-agent/output/workcell.usda"

# Verified URL from the asset catalog (--probe-s3 confirmed it resolves).
UR10E_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
)


def _rpc(tool: str, **args):
    req = urllib.request.Request(
        f"{BASE}/rpc",
        data=json.dumps({"tool": tool, "args": args}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    if "error" in body:
        raise RuntimeError(f"{tool} failed: {body['error']}")
    return body["result"]


def build() -> int:
    print("[workcell] composing benchtop workcell...")

    # 1. Table — flat slab, top at z=0.42m
    _rpc("create_primitive", prim_path="/World/workcell/table", prim_type="Cube")
    _rpc(
        "set_transform",
        prim_path="/World/workcell/table",
        translate=[0.0, 0.0, 0.40],
        scale=[0.50, 0.50, 0.02],
    )
    print("[workcell]   table       at (0, 0, 0.40), 1.0m x 1.0m x 0.04m")

    # 2. UR10e — referenced from the CDN, mounted on the table top
    _rpc(
        "add_reference_to_stage",
        usd_path=UR10E_URL,
        prim_path="/World/workcell/ur10e",
    )
    _rpc(
        "set_transform",
        prim_path="/World/workcell/ur10e",
        translate=[0.0, 0.0, 0.42],
    )
    print("[workcell]   ur10e       at (0, 0, 0.42)  [USD reference]")

    # 3. Conveyor — to the right of the table, top at z=0.425
    _rpc("create_primitive", prim_path="/World/workcell/conveyor", prim_type="Cube")
    _rpc(
        "set_transform",
        prim_path="/World/workcell/conveyor",
        translate=[1.20, 0.0, 0.40],
        scale=[0.60, 0.20, 0.025],
    )
    print("[workcell]   conveyor    at (1.20, 0, 0.40), 1.2m x 0.4m x 0.05m")

    # 4. Three microplates stacked on the conveyor
    conveyor_top = 0.40 + 0.025
    plate_half_h = 0.0075  # 1.5cm full thickness
    for i in range(3):
        path = f"/World/workcell/microplate_{i}"
        _rpc("create_primitive", prim_path=path, prim_type="Cube")
        z = conveyor_top + plate_half_h + i * (2 * plate_half_h)
        _rpc(
            "set_transform",
            prim_path=path,
            translate=[1.20, 0.0, z],
            scale=[0.06, 0.04, plate_half_h],
        )
        print(f"[workcell]   microplate_{i} at (1.20, 0, {z:.4f})")

    # 5. Verify what we just built
    sub = _rpc("query_stage", prim_path="/World/workcell", depth=1)
    children = sub["prims"][1:]
    print(f"\n[workcell] /World/workcell now has {len(children)} children:")
    for p in children:
        t = p.get("translate")
        t_str = f"  translate={t}" if t else ""
        print(f"            {p['path']}  [{p['type']}]{t_str}")

    # 6. Save
    saved = _rpc("save_stage", file_path=SAVE_PATH)
    print(f"\n[workcell] saved: {saved}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(build())
    except urllib.error.URLError as e:
        print(f"[workcell] FAILED to reach {BASE}: {e}", file=sys.stderr)
        print("[workcell] Is the container running? Try: docker compose up", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[workcell] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
