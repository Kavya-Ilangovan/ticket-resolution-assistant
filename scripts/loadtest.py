"""Tiny load test for /v1/resolve (no extra tools; works on Windows).

    python -m scripts.loadtest --url http://localhost:8000 --users 8 --requests 400

Each worker thread replays complaints from the eval sets with a random suffix so the Redis/in-memory cache is not
what you measure (use --cached to measure cache hits instead). Prints throughput, latency percentiles and errors.
"""
from __future__ import annotations

import argparse
import random
import statistics
import threading
import time

import httpx

from scripts.seed import read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--cached", action="store_true", help="repeat identical texts to measure cache hits")
    a = ap.parse_args()

    token = httpx.post(f"{a.url}/v1/auth/dev-token", params={"uid": "load", "role": "agent"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    texts = [q["text"] for q in read("eval_queries") + read("realistic_queries", "handwritten")]
    lat: list[float] = []
    errors: list[int] = []
    lock = threading.Lock()
    counter = iter(range(a.requests))

    def worker() -> None:
        with httpx.Client(timeout=60, headers=headers) as c:
            while True:
                with lock:
                    i = next(counter, None)
                if i is None:
                    return
                text = random.choice(texts) if a.cached else f"{random.choice(texts)} (ref {random.randint(0, 10**9)})"
                t0 = time.perf_counter()
                r = c.post(f"{a.url}/v1/resolve", json={"text": text})
                with lock:
                    lat.append((time.perf_counter() - t0) * 1000)
                    if r.status_code != 200:
                        errors.append(r.status_code)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(a.users)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t0
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))]  # noqa: E731
    print(f"requests={len(lat)} users={a.users} wall={wall:.1f}s throughput={len(lat) / wall:.1f} req/s errors={len(errors)} {sorted(set(errors))}")
    print(f"latency ms: mean={statistics.mean(lat):.0f} p50={q(0.5):.0f} p95={q(0.95):.0f} p99={q(0.99):.0f} max={lat[-1]:.0f}")


if __name__ == "__main__":
    main()
