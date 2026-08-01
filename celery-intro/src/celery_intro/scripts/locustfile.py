"""Locust load test for the celery-intro API.

Run with a web UI (open http://localhost:8089):
    uv run locust -f src/celery_intro/scripts/locustfile.py --host http://localhost:8000

Run headless (no UI), 50 users, spawning 10/s, for 30s:
    uv run locust -f src/celery_intro/scripts/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 30s

Start the API first:
    uv run python -m celery_intro.api.server

Note: the `/` endpoint is rate limited (5/minute), so under load it will return
429s. Those are treated as expected here (not counted as failures) so the run
reflects real throughput rather than flagging deliberate throttling as errors.
"""

from __future__ import annotations

from locust import HttpUser, between, task


class ApiUser(HttpUser):
    """Simulated user hitting the API endpoints."""

    # Each simulated user waits 1-3s between tasks.
    wait_time = between(1, 3)

    @task(3)
    def root(self) -> None:
        # `catch_response` lets us mark 429 (rate limited) as an expected outcome
        # instead of a failure, since the endpoint intentionally throttles.
        with self.client.get("/", name="/", catch_response=True) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"unexpected status {resp.status_code}")

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task(1)
    def settings(self) -> None:
        self.client.get("/settings", name="/settings")
