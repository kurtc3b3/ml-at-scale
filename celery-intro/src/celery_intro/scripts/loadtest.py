"""Load-testing CLI for the celery-intro API, built with click + rich.

Start the API first:
    uv run python -m celery_intro.api.server

Then:
    # Burst test (proves the rate limiter fires)
    uv run python -m celery_intro.scripts.loadtest burst --n 200 --concurrency 50

    # Sustained load test via Locust (headless)
    uv run python -m celery_intro.scripts.loadtest load --users 50 --spawn-rate 10 --time 30s

    # Locust with the web UI at http://localhost:8089
    uv run python -m celery_intro.scripts.loadtest load --web
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from celery_intro.scripts.test_rate_limit import burst as _burst

console = Console()

_LOCUSTFILE = Path(__file__).with_name("locustfile.py")

# Friendly labels for status codes we care about.
_STATUS_LABELS = {
    0: "transport error",
    200: "OK",
    429: "rate limited",
}


@click.group()
def cli() -> None:
    """Load-testing tools for the celery-intro API."""


@cli.command()
@click.option("--url", default="http://localhost:8000/", help="Endpoint to hit.")
@click.option("--n", default=100, show_default=True, help="Total requests to send.")
@click.option("--concurrency", default=20, show_default=True, help="Max concurrent requests.")
def burst(url: str, n: int, concurrency: int) -> None:
    """Fire a concurrent burst of requests and tally status codes."""
    with console.status(f"Firing {n} requests at {url} (concurrency {concurrency})..."):
        counts, elapsed = asyncio.run(_burst(url, n, concurrency))

    table = Table(title="Burst results", header_style="bold cyan")
    table.add_column("Status", justify="right")
    table.add_column("Label")
    table.add_column("Count", justify="right")
    for status in sorted(counts):
        label = _STATUS_LABELS.get(status, "")
        style = "green" if status == 200 else "yellow" if status == 429 else "red"
        table.add_row(f"[{style}]{status or 'ERR'}[/]", label, str(counts[status]))
    console.print(table)

    rps = n / elapsed if elapsed else 0
    limited = counts.get(429, 0)
    summary = (
        f"{n} requests in [bold]{elapsed:.2f}s[/] ([bold]{rps:.0f}[/] req/s)\n"
        f"[green]{counts.get(200, 0)}[/] OK · [yellow]{limited}[/] rate limited"
    )
    verdict = (
        "[green]Rate limiting is active.[/]"
        if limited
        else "[red]No 429s — endpoint may not have a limit applied.[/]"
    )
    console.print(Panel(f"{summary}\n\n{verdict}", title="Summary", border_style="cyan"))


@cli.command()
@click.option("--host", default="http://localhost:8000", help="Target host.")
@click.option("--users", "-u", default=50, show_default=True, help="Peak concurrent users.")
@click.option("--spawn-rate", "-r", default=10, show_default=True, help="Users spawned per second.")
@click.option("--time", "-t", "run_time", default="30s", show_default=True, help="Run duration (headless).")
@click.option("--web", is_flag=True, help="Launch the Locust web UI instead of headless.")
@click.option("--html", type=click.Path(), default=None, help="Write an HTML report to this path (headless).")
def load(host: str, users: int, spawn_rate: int, run_time: str, web: bool, html: str | None) -> None:
    """Run a sustained Locust load test against the API."""
    cmd = ["locust", "-f", str(_LOCUSTFILE), "--host", host]
    if web:
        console.print(
            Panel(
                f"Launching Locust web UI for [bold]{host}[/]\n"
                "Open [link=http://localhost:8089]http://localhost:8089[/] to drive the test.",
                title="Locust",
                border_style="cyan",
            )
        )
    else:
        cmd += ["--headless", "-u", str(users), "-r", str(spawn_rate), "-t", run_time]
        if html:
            cmd += ["--html", html]
        console.print(
            Panel(
                f"Host: [bold]{host}[/]\n"
                f"Users: [bold]{users}[/] · Spawn rate: [bold]{spawn_rate}/s[/] · Duration: [bold]{run_time}[/]"
                + (f"\nHTML report: [bold]{html}[/]" if html else ""),
                title="Locust (headless)",
                border_style="cyan",
            )
        )

    # Stream Locust's own output straight through to the terminal.
    result = subprocess.run(cmd)  # noqa: S603 - args are locally constructed, not user shell input
    sys.exit(result.returncode)


if __name__ == "__main__":
    cli()
