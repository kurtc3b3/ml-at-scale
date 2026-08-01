"""Fire a burst of concurrent requests at the API to exercise rate limiting.

Usage:
    uv run python -m celery_intro.scripts.test_rate_limit
    uv run python -m celery_intro.scripts.test_rate_limit --url http://localhost:8000/ --n 100 --concurrency 20

Rate limiting only triggers on routes decorated with `@limiter.limit(...)` in
the server. If nothing returns 429, check that the target endpoint has a limit
applied.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

import httpx


async def _one(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore) -> int:
    """Send a single request, returning its HTTP status (0 on transport error)."""
    async with sem:
        try:
            resp = await client.get(url)
            return resp.status_code
        except httpx.HTTPError:
            return 0


async def burst(url: str, n: int, concurrency: int) -> tuple[Counter, float]:
    """Fire `n` requests at `url`, `concurrency` at a time.

    Returns a Counter of status codes (0 == transport error) and elapsed seconds.
    """
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=10.0) as client:
        start = time.perf_counter()
        results = await asyncio.gather(*(_one(client, url, sem) for _ in range(n)))
        elapsed = time.perf_counter() - start
    return Counter(results), elapsed


async def run(url: str, n: int, concurrency: int) -> None:
    counts, elapsed = await burst(url, n, concurrency)
    ok = counts.get(200, 0)
    limited = counts.get(429, 0)

    print(f"Target:       {url}")
    print(f"Requests:     {n} at concurrency {concurrency}")
    print(f"Elapsed:      {elapsed:.2f}s  ({n / elapsed:.0f} req/s)")
    print("Status codes:")
    for status in sorted(counts):
        label = {0: "transport error"}.get(status, "")
        print(f"  {status or 'ERR':<4} {label:<16} {counts[status]}")
    print(f"\n200 OK:       {ok}")
    print(f"429 limited:  {limited}")
    if limited:
        print("Rate limiting is active.")
    else:
        print("No 429s — the endpoint may not have a limit applied.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test API rate limiting.")
    parser.add_argument("--url", default="http://localhost:8000/", help="Endpoint to hit.")
    parser.add_argument("--n", type=int, default=100, help="Total requests to send.")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests.")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.n, args.concurrency))


if __name__ == "__main__":
    main()
