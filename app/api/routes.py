"""FastAPI route handlers for the Language Feedback API.

Endpoints:
  GET  /health           — liveness probe for Docker / load balancers.
  POST /feedback         — single-sentence correction feedback.
  POST /feedback/batch   — multi-sentence batch analysis (1–10 sentences).
  GET  /stats            — real-time cache and in-flight telemetry.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_llm_service
from app.models.schemas import (
    BatchFeedbackItem,
    BatchFeedbackRequest,
    BatchFeedbackResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from app.services import cache as cache_service
from app.services import llm as llm_module
from app.services.llm import LLMService

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Ops ───────────────────────────────────────────────────────────────────────


@router.get("/health", tags=["ops"])
async def health_check() -> dict[str, str]:
    """Liveness probe — confirms the server process is running.

    Returns:
        ``{"status": "ok"}``
    """
    return {"status": "ok"}


@router.get("/stats", tags=["ops"])
async def stats() -> dict:
    """Real-time telemetry for cache efficiency and in-flight requests.

    Useful for validating that caching is active (hit_rate should rise during
    load tests with repeated sentences) and that in-flight deduplication is
    working (in_flight_requests should stay low under concurrent identical load).

    Returns:
        Dict with cache_size, hits, misses, hit_rate, and in_flight_requests.
    """
    data = cache_service.get_stats()
    # Expose the number of LLM calls currently awaiting a response.
    data["in_flight_requests"] = llm_module.get_in_flight_count()
    return data


# ── Core feedback ─────────────────────────────────────────────────────────────


@router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def feedback(
    request: FeedbackRequest,
    service: LLMService = Depends(get_llm_service),
) -> FeedbackResponse:
    """Analyse a single learner sentence and return structured correction feedback.

    Benefits from two-layer deduplication:
      - Persistent cache: repeated identical requests cost zero LLM tokens.
      - In-flight dedup: concurrent identical requests share one LLM call.

    Args:
        request: Validated request body (sentence + language pair).
        service: Injected LLMService singleton.

    Returns:
        FeedbackResponse with corrected sentence, error list, and CEFR rating.

    Raises:
        HTTPException 422: Pydantic validation failure on the request body.
        HTTPException 500: Unrecoverable LLM or internal error.
    """
    try:
        return await service.get_feedback(request)
    except Exception as exc:
        logger.exception("Unhandled error in POST /feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post(
    "/feedback/batch",
    response_model=BatchFeedbackResponse,
    tags=["feedback"],
)
async def feedback_batch(
    request: BatchFeedbackRequest,
    service: LLMService = Depends(get_llm_service),
) -> BatchFeedbackResponse:
    """Analyse 1–10 sentences in the same language pair concurrently.

    All sentences share ``target_language`` and ``native_language``.
    Requests are issued concurrently via ``asyncio.gather`` so the total
    latency is close to the slowest individual sentence, not the sum.

    Cached sentences return instantly and don't count against the model budget.
    Failed sentences return a non-null ``error`` field rather than aborting
    the whole batch — partial success is better than all-or-nothing for
    classroom use cases where most sentences may be correct.

    Args:
        request: BatchFeedbackRequest with 1–10 sentences.
        service: Injected LLMService singleton.

    Returns:
        BatchFeedbackResponse with per-sentence results and summary counts.

    Raises:
        HTTPException 422: Validation failure (too many sentences, empty list).
    """
    # Build individual FeedbackRequest objects for each sentence.
    individual_requests = [
        FeedbackRequest(
            sentence=sentence,
            target_language=request.target_language,
            native_language=request.native_language,
        )
        for sentence in request.sentences
    ]

    # Issue all calls concurrently; return_exceptions=True means one failure
    # doesn't cancel the rest of the batch.
    raw_results = await asyncio.gather(
        *[service.get_feedback(req) for req in individual_requests],
        return_exceptions=True,
    )

    items = [
        BatchFeedbackItem(
            sentence=sentence,
            result=result if not isinstance(result, BaseException) else None,
            error=str(result) if isinstance(result, BaseException) else None,
        )
        for sentence, result in zip(request.sentences, raw_results)
    ]

    return BatchFeedbackResponse(
        results=items,
        total=len(items),
        succeeded=sum(1 for item in items if item.result is not None),
        failed=sum(1 for item in items if item.error is not None),
    )
