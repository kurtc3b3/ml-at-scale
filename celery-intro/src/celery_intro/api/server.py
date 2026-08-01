from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler

from structlog import get_logger

logger = get_logger()

logger.info("Logging initialized", message="Starting the application")

limiter = Limiter(key_func=lambda: "global")

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.get("/")
@limiter.limit("5/minute")
def root(request: Request):
    return {"message": "Hello World!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/settings")
def read_settings():
    from celery_intro.settings import get_settings
    settings = get_settings()
    return {
        "app_name": settings.app_name,
        "debug": settings.debug,
        "database_url": settings.database_url,
    }


@app.post("/add")
def enqueue_add(x: int, y: int):
    """Enqueue the Celery `add` task and return its id (requires a running worker)."""
    from celery_intro.proc.celery_example import add

    task = add.delay(x, y)
    return {"task_id": task.id, "status": "queued"}


@app.get("/add/{task_id}")
def get_add_result(task_id: str):
    """Fetch the status/result of a previously enqueued `add` task."""
    from celery_intro.proc.celery_example import app as celery_app

    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
