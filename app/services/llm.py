"""Anthropic LLM service: tool-use outputs, retry, caching, and in-flight dedup.

Key design decisions:
  1. Tool-use + tool_choice={"type":"tool"} forces the model to always call the
     language_feedback tool — mathematically guaranteed schema compliance.

  2. Turn-based few-shot examples are prepended to every messages list.
     This is the correct technique for Anthropic tool-use: the model sees
     real tool_use blocks in prior turns, not schema descriptions in text.

  3. tenacity wraps the raw API call with exponential back-off for the four
     transient error types (rate-limit, connection, timeout, server-error).

  4. The Anthropic async client is a module-level singleton — HTTP connection
     pools are reused across requests.

  5. Two-layer deduplication:
       Layer 1 — persistent TTL cache: zero tokens on cache hit.
       Layer 2 — in-flight Future map: if N concurrent requests for the SAME
                 sentence arrive before the first completes, only ONE LLM call
                 fires; all N waiters share its result via asyncio.shield().
                 This prevents the "thundering herd" problem on popular
                 exercises (e.g. a whole class submitting the same sentence).
"""

import asyncio
import logging
from asyncio import Future
from typing import Any

import anthropic
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.models.schemas import ErrorDetail, FeedbackRequest, FeedbackResponse
from app.services import cache as cache_service
from app.services.prompts import (
    ACTIVE_FEW_SHOT_LAST_TOOL_USE_ID,
    ACTIVE_FEW_SHOT_PREFIX,
    ACTIVE_SYSTEM_PROMPT,
    ACTIVE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# ── Enum normalisation ────────────────────────────────────────────────────────
# Anthropic's tool-use schema guides (but does not strictly enforce) enum values.
# If the model returns an unrecognised error_type (e.g. "case", "tense"),
# we normalise it to "other" rather than raising a ValidationError that would
# discard otherwise valid feedback.
_VALID_ERROR_TYPES = frozenset(
    [
        "grammar", "spelling", "word_choice", "punctuation", "word_order",
        "missing_word", "extra_word", "conjugation", "gender_agreement",
        "number_agreement", "tone_register", "other",
    ]
)


def _normalise_error(raw_error: dict) -> dict:
    """Return raw_error with error_type coerced to a valid enum value.

    Args:
        raw_error: Dict from the LLM tool_use block representing one error.

    Returns:
        The same dict, with error_type guaranteed to be in _VALID_ERROR_TYPES.
    """
    error_type = raw_error.get("error_type", "other")
    if error_type not in _VALID_ERROR_TYPES:
        logger.warning("LLM returned unknown error_type %r — remapping to 'other'", error_type)
        raw_error = {**raw_error, "error_type": "other"}
    return raw_error


# ── Singleton Anthropic async client ─────────────────────────────────────────
_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Return (or lazily initialise) the shared Anthropic async client."""
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.request_timeout,
        )
    return _client


# ── In-flight request deduplication ──────────────────────────────────────────
# Maps cache_key → asyncio.Future[FeedbackResponse].
#
# When two or more concurrent requests for the exact same sentence arrive:
#   - The first request creates a Future and begins the LLM call.
#   - Subsequent requests find the key in _in_flight and await the same Future.
#   - When the first request resolves, ALL waiters receive the result instantly.
#
# asyncio is single-threaded, so the check `if key in _in_flight` followed by
# `_in_flight[key] = future` is effectively atomic (no await between them).
_in_flight: dict[str, Future[FeedbackResponse]] = {}


def get_in_flight_count() -> int:
    """Return the number of LLM calls currently in progress.

    Used by the /stats endpoint for real-time observability.

    Returns:
        Number of active in-flight requests.
    """
    return len(_in_flight)


# ── Tool definition ───────────────────────────────────────────────────────────
_FEEDBACK_TOOL: dict[str, Any] = {
    "name": "language_feedback",
    "description": "Return structured language feedback for a learner's sentence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "corrected_sentence": {
                "type": "string",
                "description": (
                    "Minimally corrected sentence. "
                    "Must be identical to the input when the sentence has no errors."
                ),
            },
            "is_correct": {
                "type": "boolean",
                "description": "True only when the sentence has zero errors.",
            },
            "errors": {
                "type": "array",
                "description": "Empty array when is_correct is true.",
                "items": {
                    "type": "object",
                    "properties": {
                        "original": {
                            "type": "string",
                            "description": "Erroneous word or phrase from the original sentence.",
                        },
                        "correction": {"type": "string", "description": "Corrected replacement."},
                        "error_type": {
                            "type": "string",
                            "enum": list(_VALID_ERROR_TYPES),
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation written in the learner's native language.",
                        },
                    },
                    "required": ["original", "correction", "error_type", "explanation"],
                },
            },
            "difficulty": {
                "type": "string",
                "enum": ["A1", "A2", "B1", "B2", "C1", "C2"],
                "description": "CEFR complexity rating of the input sentence.",
            },
        },
        "required": ["corrected_sentence", "is_correct", "errors", "difficulty"],
    },
}

# ── Retry-wrapped API call ────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(
        (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        )
    ),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _call_anthropic(
    system: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Send a request to the Anthropic API and extract the tool_use block.

    The messages list should already include any few-shot examples followed
    by the actual user message — this function is purely transport.

    Args:
        system: System prompt string.
        messages: Full messages list (few-shot turns + user request).

    Returns:
        The raw dict from the ``tool_use`` block's ``input`` field.

    Raises:
        ValueError: If the response contains no ``tool_use`` block.
        anthropic.*Error: Re-raised after all retry attempts are exhausted.
    """
    client = _get_client()
    response = await client.messages.create(
        model=settings.model,
        # 1024 tokens covers any single-sentence feedback with room to spare.
        # Keeping this low minimises worst-case latency and cost per request.
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=[_FEEDBACK_TOOL],
        # Forcing tool use means the model can never fall back to plain text.
        tool_choice={"type": "tool", "name": "language_feedback"},
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input  # type: ignore[return-value]

    raise ValueError(
        "Anthropic response contained no tool_use block — check tool_choice config"
    )


# ── Service class ─────────────────────────────────────────────────────────────


class LLMService:
    """Orchestrates few-shot prompting, LLM calls, deduplication, and caching.

    Stateless by design — safe to share as a singleton across all concurrent
    requests via FastAPI's dependency injection.
    """

    async def get_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Return structured language feedback with two-layer deduplication.

        Request flow:
          1. Compute deterministic cache key.
          2. Layer 1 — cache hit: return stored result (0 ms, 0 tokens).
          3. Layer 2 — in-flight hit: await the Future from an identical
             concurrent request (0 extra tokens, minimal latency).
          4. Miss: build few-shot messages, call Anthropic, parse, cache.

        Args:
            request: Validated FeedbackRequest from the API layer.

        Returns:
            FeedbackResponse conforming to the JSON schema.

        Raises:
            anthropic.*Error: If all retry attempts are exhausted.
        """
        cache_key = cache_service.make_cache_key(
            request.sentence,
            request.target_language,
            request.native_language,
        )

        # ── Layer 1: persistent cache ─────────────────────────────────────────
        cached = cache_service.get_cached(cache_key)
        if cached is not None:
            logger.info("Cache hit [key=%s…]", cache_key[:8])
            return cached

        # ── Layer 2: in-flight deduplication ─────────────────────────────────
        # No await between the check and the assignment, so this is atomic
        # under asyncio's single-threaded execution model.
        if cache_key in _in_flight:
            logger.info("In-flight dedup hit [key=%s…]", cache_key[:8])
            # asyncio.shield ensures that if THIS coroutine is cancelled, the
            # underlying Future (owned by the first request) keeps running and
            # other waiters still receive their result.
            return await asyncio.shield(_in_flight[cache_key])

        loop = asyncio.get_running_loop()
        future: Future[FeedbackResponse] = loop.create_future()
        _in_flight[cache_key] = future

        try:
            user_message = ACTIVE_USER_PROMPT.format(
                sentence=request.sentence,
                target_language=request.target_language,
                native_language=request.native_language,
            )
            # Build the full messages list:
            #   FEW_SHOT_PREFIX  (examples A and B, ending with assistant tool_use)
            #   + one merged user turn containing:
            #       - the tool_result for the last few-shot tool_use (required
            #         by Anthropic: every tool_use must be followed by a result)
            #       - the actual user request as a text block
            # This avoids consecutive user messages while satisfying the API rule.
            messages = ACTIVE_FEW_SHOT_PREFIX + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": ACTIVE_FEW_SHOT_LAST_TOOL_USE_ID,
                            "content": "Feedback recorded.",
                        },
                        {"type": "text", "text": user_message},
                    ],
                }
            ]

            raw = await _call_anthropic(ACTIVE_SYSTEM_PROMPT, messages)

            # Normalise error_type before Pydantic validation to handle any
            # enum values the model returns that aren't in our allowed list.
            response = FeedbackResponse(
                corrected_sentence=raw["corrected_sentence"],
                is_correct=raw["is_correct"],
                errors=[ErrorDetail(**_normalise_error(e)) for e in raw.get("errors", [])],
                difficulty=raw["difficulty"],
            )

            cache_service.set_cached(cache_key, response)
            # Resolve the Future so all waiters from Layer 2 receive the result.
            future.set_result(response)

            logger.info(
                "LLM call complete [is_correct=%s, errors=%d, difficulty=%s]",
                response.is_correct,
                len(response.errors),
                response.difficulty,
            )
            return response

        except Exception as exc:
            # Propagate the exception to all Layer-2 waiters so they fail fast
            # rather than hanging indefinitely.
            if not future.done():
                future.set_exception(exc)
            raise

        finally:
            # Always remove the Future from the map so subsequent requests
            # (after this one completes) start fresh and hit the cache.
            _in_flight.pop(cache_key, None)
