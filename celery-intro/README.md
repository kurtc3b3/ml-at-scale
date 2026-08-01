# Celery Intro

A learning project exploring how to build and operate ML-serving APIs at scale — a
FastAPI service with rate limiting, plus load-testing tooling.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
uv sync
```

Optionally create a `.env` from the template to override settings:

```bash
cp .env.template .env
```

Settings (see [`settings.py`](src/celery_intro/settings.py)) are read from the
environment / `.env`:

| Setting        | Env var        | Default                  |
| -------------- | -------------- | ------------------------ |
| `app_name`     | `APP_NAME`     | `Celery Intro`           |
| `debug`        | `DEBUG`        | `False`                  |
| `database_url` | `DATABASE_URL` | `sqlite:///./test.db`    |

## Running the API

```bash
uv run python -m celery_intro.api.server
```

Serves on `http://localhost:8000`.

### Endpoints

| Method | Path             | Notes                                              |
| ------ | ---------------- | -------------------------------------------------- |
| GET    | `/`              | Hello World — rate limited `5/min`                 |
| GET    | `/health`        | Health check                                       |
| GET    | `/settings`      | Returns current settings                           |
| POST   | `/add?x=&y=`     | Enqueue the Celery `add` task, returns a `task_id` |
| GET    | `/add/{task_id}` | Fetch a task's `status` and `result`               |

The `/add` endpoints require a running Celery worker + Redis (see
[Celery + Flower](#celery--flower--task-queue)). Example:

```bash
curl -X POST "http://localhost:8000/add?x=4&y=6"   # -> {"task_id":"...","status":"queued"}
curl "http://localhost:8000/add/<task_id>"          # -> {"status":"SUCCESS","result":10}
```

Rate limiting uses [slowapi](https://github.com/laurentS/slowapi). The `/`
endpoint is limited to `5/minute` and returns `429 Too Many Requests` when
exceeded. The limiter currently keys on a single global bucket
(`key_func=lambda: "global"`); swap in `slowapi.util.get_remote_address` for
per-client limits.

## Load testing

An `celery-loadtest` CLI (built with [click](https://click.palletsprojects.com/) +
[rich](https://rich.readthedocs.io/)) wraps two tools. **Start the API first**,
then in another shell:

### Burst test — prove the rate limiter fires

Fires a concurrent burst via `httpx.AsyncClient` and tallies status codes.

```bash
uv run celery-loadtest burst --n 200 --concurrency 50
```

Expect ~5 `200`s and the rest `429`s against `/`.

### Sustained load test — Locust

```bash
# headless, with an optional HTML report
uv run celery-loadtest load --users 50 --spawn-rate 10 --time 30s --html report.html

# or the Locust web UI at http://localhost:8089
uv run celery-loadtest load --web
```

The Locust user hits `/`, `/health`, and `/settings` with a 3:2:1 weighting.
`429`s on `/` are treated as expected (not failures), so results reflect real
throughput rather than flagging deliberate throttling.

You can also run the underlying scripts directly:

```bash
uv run python -m celery_intro.scripts.test_rate_limit --url http://localhost:8000/ --n 200 --concurrency 50
uv run locust -f src/celery_intro/scripts/locustfile.py --host http://localhost:8000
```

## Celery + Flower — task queue

Requires a running **Redis** (broker + backend):

```bash
docker compose up -d
redis-cli ping   # -> PONG
```

Or run Redis any other way on `localhost:6379` (see
[`celery_example.py`](src/celery_intro/proc/celery_example.py)).

Start a worker (in one shell). `-E` emits task events so Flower shows history:

```bash
uv run celery -A celery_intro.proc.celery_example worker --loglevel=info -E
```

Dispatch a task (in another shell):

```bash
uv run python -c "from celery_intro.proc.celery_example import add; print(add.delay(4, 6).get(timeout=10))"
# -> 10
```

Launch the Flower monitoring UI at http://localhost:5555:

```bash
uv run celery -A celery_intro.proc.celery_example flower --port=5555
```

## Project layout

```
src/celery_intro/
├── api/
│   └── server.py          # FastAPI app + slowapi rate limiting
├── proc/
│   └── celery_example.py  # Celery task (Redis broker) + Flower UI
├── scripts/
│   ├── loadtest.py        # click + rich CLI (celery-loadtest)
│   ├── locustfile.py      # Locust load test
│   └── test_rate_limit.py # httpx burst test
└── settings.py            # pydantic-settings config
```
