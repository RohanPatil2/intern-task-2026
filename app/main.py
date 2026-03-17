"""FastAPI application entry point.

Responsibilities here are intentionally minimal:
  - Create the FastAPI instance with metadata.
  - Register the router from app.api.routes.
  - Validate required config at startup via the lifespan hook.

All business logic lives in app/services/, all routes in app/api/routes.py.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup checks before the server accepts requests.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is missing — fail fast rather than
            accepting requests that will immediately fail at the LLM call.
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
        "POST /feedback to analyse a sentence; GET /health for a liveness probe."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
