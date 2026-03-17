"""Anthropic LLM service: structured tool-use outputs, retry logic, and caching.

Key design decisions:
  1. Tool-use with tool_choice={"type":"tool"} forces the model to *always*
     call the language_feedback tool, giving us mathematically guaranteed
     JSON schema compliance — no raw-string parsing, no json.loads guesswork.

  2. tenacity wraps the raw API call with exponential back-off so transient
     rate-limit or connection errors are retried transparently before the
     caller ever sees an exception.

  3. The Anthropic async client is a module-level singleton so HTTP connection
     pools are reused across requests, keeping per-call latency low.

  4. All cached lookups happen *before* the LLM call, so a cache hit costs
     ~0 ms and zero tokens.
"""

import logging
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
from app.services.prompts import ACTIVE_SYSTEM_PROMPT, ACTIVE_USER_PROMPT

logger = logging.getLogger(__name__)

# ── Singleton Anthropic async client ─────────────────────────────────────────
# One client per process; the underlying httpx.AsyncClient handles connection
# pooling automatically.
_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Return (or lazily initialise) the shared Anthropic async client.

    Returns:
        The module-level AsyncAnthropic instance.
    """
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.request_timeout,
        )
    return _client


# ── Tool definition ───────────────────────────────────────────────────────────
# Defining the tool schema here (rather than inline in the API call) makes it
# easy to version and review alongside the prompt templates.
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
                        "correction": {
                            "type": "string",
                            "description": "Corrected replacement.",
                        },
                        "error_type": {
                            "type": "string",
                            "enum": [
                                "grammar",
                                "spelling",
                                "word_choice",
                                "punctuation",
                                "word_order",
                                "missing_word",
                                "extra_word",
                                "conjugation",
                                "gender_agreement",
                                "number_agreement",
                                "tone_register",
                                "other",
                            ],
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
# Retry on the four exception types that are safe to retry (all are transient):
#   - RateLimitError  : 429, back off and retry
#   - APIConnectionError: DNS / TCP failure, retry
#   - APITimeoutError : request timed out at the HTTP level, retry
#   - InternalServerError: Anthropic 500, retry
# Any other exception (AuthenticationError, BadRequestError, etc.) propagates
# immediately — retrying those would waste tokens and time.
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
async def _call_anthropic(system: str, user_content: str) -> dict[str, Any]:
    """Send a request to the Anthropic API and extract the tool_use block.

    Args:
        system: System prompt string.
        user_content: Formatted user message string.

    Returns:
        The raw dict from the ``tool_use`` block's ``input`` field.

    Raises:
        ValueError: If the response contains no ``tool_use`` block
            (should be unreachable given tool_choice enforcement).
        anthropic.*Error: Re-raised after all retry attempts are exhausted.
    """
    client = _get_client()
    response = await client.messages.create(
        model=settings.model,
        # 1024 tokens is ample for structured feedback on a single sentence;
        # keeping max_tokens low reduces worst-case latency and cost.
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        tools=[_FEEDBACK_TOOL],
        # tool_choice={"type":"tool"} forces the model to call language_feedback
        # on every request — it cannot emit a plain-text response.
        tool_choice={"type": "tool", "name": "language_feedback"},
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input  # type: ignore[return-value]

    # This branch is unreachable under normal operation but provides a clear
    # error message if Anthropic ever changes its tool_choice behaviour.
    raise ValueError("Anthropic response contained no tool_use block — check tool_choice config")


# ── Service class ─────────────────────────────────────────────────────────────


class LLMService:
    """Orchestrates prompt construction, LLM calls, caching, and response parsing.

    This class is intentionally stateless so it is safe to share as a singleton
    across all concurrent requests via FastAPI's dependency injection system.
    """

    async def get_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Return structured language feedback, hitting the cache when possible.

        Flow:
          1. Compute deterministic cache key.
          2. Return cached result immediately on a hit (zero LLM tokens used).
          3. On a miss: build prompts, call Anthropic with retry logic,
             parse the tool_use output through Pydantic, store in cache.

        Args:
            request: Validated FeedbackRequest from the API layer.

        Returns:
            A FeedbackResponse conforming to the JSON schema.

        Raises:
            pydantic.ValidationError: If the LLM returns data that violates
                the schema constraints (e.g. an unknown error_type).
            anthropic.*Error: If all retry attempts fail.
        """
        cache_key = cache_service.make_cache_key(
            request.sentence,
            request.target_language,
            request.native_language,
        )

        # Fast path — no network call, no tokens consumed.
        cached = cache_service.get_cached(cache_key)
        if cached is not None:
            logger.info("Cache hit [key=%s…]", cache_key[:8])
            return cached

        system_prompt = ACTIVE_SYSTEM_PROMPT
        user_message = ACTIVE_USER_PROMPT.format(
            sentence=request.sentence,
            target_language=request.target_language,
            native_language=request.native_language,
        )

        raw = await _call_anthropic(system_prompt, user_message)

        # Pydantic validates enum membership for error_type and difficulty,
        # catching any schema drift before it reaches the caller.
        response = FeedbackResponse(
            corrected_sentence=raw["corrected_sentence"],
            is_correct=raw["is_correct"],
            errors=[ErrorDetail(**e) for e in raw.get("errors", [])],
            difficulty=raw["difficulty"],
        )

        cache_service.set_cached(cache_key, response)
        logger.info(
            "LLM call complete [is_correct=%s, errors=%d, difficulty=%s]",
            response.is_correct,
            len(response.errors),
            response.difficulty,
        )
        return response
