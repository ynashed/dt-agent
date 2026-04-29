"""
sim_client_smoke.py — Validate the HTTP RPC pipe to the Isaac Sim container.

Runs on the host. Lists available tools, calls get_stage_info, and prints
the result. Uses stdlib urllib so it works without installing any host-side
deps for this validator.

    python scripts/sim_client_smoke.py
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8765"


def main() -> int:
    try:
        with urllib.request.urlopen(f"{BASE}/tools", timeout=10) as r:
            tools = json.loads(r.read())
        print(f"[smoke] tools: {tools}")

        req = urllib.request.Request(
            f"{BASE}/rpc",
            data=json.dumps({"tool": "get_stage_info"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
        print(f"[smoke] get_stage_info: {result}")
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
