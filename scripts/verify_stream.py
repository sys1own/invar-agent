"""Verifies that /api/agent/stream yields valid SSE events without blocking."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"


def main() -> int:
    url = f"{BASE}/api/agent/stream?objective=verify%20sse%20streaming"
    start = time.perf_counter()
    first_event_at: float | None = None
    events: list[dict] = []
    with urllib.request.urlopen(url, timeout=30) as resp:
        assert resp.headers["Content-Type"].startswith("text/event-stream"), resp.headers
        buffer = ""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buffer += chunk.decode("utf-8")
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                for line in frame.splitlines():
                    if line.startswith("data: "):
                        payload = json.loads(line[6:])
                        if first_event_at is None:
                            first_event_at = time.perf_counter() - start
                        events.append(payload)
            if any(e.get("type") == "done" for e in events):
                break
    elapsed = time.perf_counter() - start
    types = [e.get("type") for e in events]
    assert "stage" in types, f"missing stage events: {types}"
    assert "done" in types, f"missing done event: {types}"
    assert first_event_at is not None and first_event_at < 5.0, f"first event too slow: {first_event_at}"
    print(f"OK: {len(events)} SSE events, first at {first_event_at:.2f}s, total {elapsed:.2f}s")
    print("event types:", types)
    return 0


if __name__ == "__main__":
    sys.exit(main())
