"""FastAPI route handlers for the Language Feedback API.

Endpoints:
  GET  /health    — liveness probe used by Docker Compose and load balancers.
  POST /feedback  — core endpoint: analyse a sentence, return structured feedback.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_llm_service
from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.services.llm import LLMService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["ops"])
async def health_check() -> dict[str, str]:
    """Liveness probe — confirms the server is running and reachable.

    Returns:
        A JSON object ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def feedback(
    request: FeedbackRequest,
    service: LLMService = Depends(get_llm_service),
) -> FeedbackResponse:
    """Analyse a learner's sentence and return structured correction feedback.

    The response is pulled from an in-memory cache on repeated identical
    requests, so subsequent calls for the same sentence cost zero LLM tokens.

    Args:
        request: Validated request body containing sentence and language fields.
        service: Injected LLMService singleton (see core/dependencies.py).

    Returns:
        FeedbackResponse with corrected sentence, error list, and CEFR rating.

    Raises:
        HTTPException 422: Pydantic validation failure on the request body.
        HTTPException 500: Unrecoverable LLM or internal error.
    """
    try:
        return await service.get_feedback(request)
    except Exception as exc:
        # Log the full traceback for observability, then return a clean 500
        # rather than leaking internal details to the client.
        logger.exception("Unhandled error processing /feedback request: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
