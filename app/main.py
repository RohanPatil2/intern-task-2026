"""FastAPI application entry point.

Responsibilities:
  - Create the FastAPI instance with metadata.
  - Attach correlation-ID + timing middleware.
  - Register all routes from app.api.routes.
  - Validate required configuration at startup via the lifespan hook.

All business logic lives in app/services/; all route handlers in app/api/routes.py.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup checks before accepting requests.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is not configured — fail fast rather
            than accepting requests that will immediately fail at the LLM call.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    logger.info("Starting Language Feedback API (model=%s)", settings.model)
    yield
    logger.info("Language Feedback API shutting down")


app = FastAPI(
    title="Language Feedback API",
    description=(
        "LLM-powered grammar correction for language learners. "
        "Supports single-sentence analysis (POST /feedback), "
        "batch analysis (POST /feedback/batch), "
        "and real-time telemetry (GET /stats)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests so the API can be called from a browser-based
# frontend (e.g. the Pangea Chat web app) without a proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_and_timing(request: Request, call_next) -> Response:
    """Attach a correlation ID and response-time header to every response.

    The correlation ID is taken from the incoming X-Request-ID header if
    present (so clients can trace a request end-to-end), otherwise a new
    UUID-4 is generated.

    Headers added to every response:
      X-Request-ID      — UUID-4 string, echoed back or freshly generated.
      X-Response-Time-Ms — Wall-clock time in milliseconds for the full request.

    Args:
        request: Incoming HTTP request.
        call_next: Next middleware or route handler in the stack.

    Returns:
        The response with the two headers added.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

    logger.info(
        "%s %s → %d  [%d ms]  request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


app.include_router(router)
