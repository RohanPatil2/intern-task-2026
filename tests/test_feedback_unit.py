"""Unit tests for the language feedback service and API endpoints.

All Anthropic API calls are mocked — no API key required.  These tests run
in milliseconds and cover:

  Core service:
    - Spanish conjugation error parsing
    - Correct sentence edge case (is_correct=True, empty errors)
    - French multi-error detection
    - Japanese CJK script round-trip
    - Persistent cache hit (LLM called once, second response from cache)
    - Portuguese mixed error types
    - Native-language explanation preservation

  Input validation:
    - Invisible Unicode control characters stripped cleanly
    - Sentence exceeding 2000-char limit rejected with 422

  Few-shot:
    - Few-shot messages are included in the messages list sent to the LLM

  Concurrency:
    - In-flight deduplication (two concurrent identical requests fire ONE LLM call)

  Batch endpoint:
    - Happy-path batch with two sentences
    - Partial-failure batch (one succeeds, one fails) returns both results
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.services.llm import LLMService

# ── Shared test client ────────────────────────────────────────────────────────
client = TestClient(app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_request(
    sentence: str, target: str = "Spanish", native: str = "English"
) -> FeedbackRequest:
    """Convenience factory for FeedbackRequest objects."""
    return FeedbackRequest(sentence=sentence, target_language=target, native_language=native)


def _correct_raw(sentence: str, difficulty: str = "A1") -> dict:
    """Return mock LLM output for a correct sentence."""
    return {
        "corrected_sentence": sentence,
        "is_correct": True,
        "errors": [],
        "difficulty": difficulty,
    }


# ── Core service tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spanish_conjugation_error_parsed_correctly():
    """Verifies a Spanish conjugation error is detected and all response fields
    (corrected_sentence, is_correct, error_type, difficulty) are populated
    correctly from the mocked tool_use output.

    'Yo soy fue al mercado' mixes 'soy' (present of ser) with 'fue' (past of
    ir); the correct preterite is 'fui'.
    """
    mock_raw = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms. Use 'fui' (I went).",
            }
        ],
        "difficulty": "A2",
    }
    with patch("app.services.llm._call_anthropic", new=AsyncMock(return_value=mock_raw)):
        result = await LLMService().get_feedback(_make_request("Yo soy fue al mercado ayer."))

    assert isinstance(result, FeedbackResponse)
    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.errors[0].original == "soy fue"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_correct_sentence_returns_empty_errors_and_original():
    """Verifies the 'already correct' edge case: is_correct must be True,
    errors must be an empty list, and corrected_sentence must equal the input.

    This is explicitly required by the task spec and worth isolating to catch
    prompt regressions that cause the model to hallucinate errors.
    """
    sentence = "Ich habe gestern einen interessanten Film gesehen."
    with patch(
        "app.services.llm._call_anthropic",
        new=AsyncMock(return_value=_correct_raw(sentence, "B1")),
    ):
        result = await LLMService().get_feedback(_make_request(sentence, target="German"))

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == sentence


@pytest.mark.asyncio
async def test_french_two_gender_agreement_errors_returned():
    """Verifies that two simultaneous gender agreement errors in French are
    both detected and classified correctly.

    'La chat mange le table':
      - 'La chat'  → 'Le chat'  (chat is masculine)
      - 'le table' → 'la table' (table is feminine)
    """
    mock_raw = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine — use 'le'.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine — use 'la'.",
            },
        ],
        "difficulty": "A1",
    }
    with patch("app.services.llm._call_anthropic", new=AsyncMock(return_value=mock_raw)):
        result = await LLMService().get_feedback(
            _make_request("La chat noir est sur le table.", target="French")
        )

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


@pytest.mark.asyncio
async def test_japanese_non_latin_script_handled_without_encoding_errors():
    """Verifies Japanese (CJK) input and output round-trips cleanly with no
    character encoding errors.

    '図書館を住んでいます' uses the wrong particle 'を'; the location particle
    for 住む (to live) is 'に'.
    """
    mock_raw = {
        "corrected_sentence": "図書館に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "Use 'に' for location with 住む.",
            }
        ],
        "difficulty": "A2",
    }
    with patch("app.services.llm._call_anthropic", new=AsyncMock(return_value=mock_raw)):
        result = await LLMService().get_feedback(
            _make_request("図書館を住んでいます。", target="Japanese")
        )

    assert result.is_correct is False
    assert "に" in result.corrected_sentence


@pytest.mark.asyncio
async def test_cache_hit_skips_llm_on_second_identical_request():
    """Verifies the persistent cache prevents duplicate LLM calls.

    First request should call _call_anthropic exactly once; the second
    identical request must be served from cache with zero additional calls.
    """
    sentence = "Je mange une pomme chaque matin."
    request = _make_request(sentence, target="French")

    from app.services.cache import _cache, _lock

    with _lock:
        _cache.clear()

    call_count = 0

    async def counting_mock(*args, **kwargs) -> dict:
        nonlocal call_count
        call_count += 1
        return _correct_raw(sentence)

    with patch("app.services.llm._call_anthropic", new=counting_mock):
        service = LLMService()
        first = await service.get_feedback(request)
        second = await service.get_feedback(request)

    assert call_count == 1, "LLM must only be called once; second response should come from cache"
    assert first == second


@pytest.mark.asyncio
async def test_portuguese_mixed_spelling_and_grammar_errors():
    """Verifies a sentence with multiple error types (spelling + grammar)
    returns all errors with correct categories.
    """
    mock_raw = {
        "corrected_sentence": "Eu quero comprar um presente para minha irmã.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "Correct spelling is 'presente'.",
            },
            {
                "original": "pra",
                "correction": "para",
                "error_type": "tone_register",
                "explanation": "'Pra' is informal; 'para' is standard.",
            },
        ],
        "difficulty": "B1",
    }
    with patch("app.services.llm._call_anthropic", new=AsyncMock(return_value=mock_raw)):
        result = await LLMService().get_feedback(
            _make_request("Eu quero comprar um prezente pra minha irmã.", target="Portuguese")
        )

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert {e.error_type for e in result.errors} == {"spelling", "tone_register"}


@pytest.mark.asyncio
async def test_explanation_language_field_is_preserved():
    """Verifies that explanations are stored and returned verbatim, confirming
    the native_language field drives explanation content for non-English learners.
    """
    spanish_explanation = "El artículo 'la' es femenino, pero 'chat' es masculino."
    mock_raw = {
        "corrected_sentence": "Le chat est mignon.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": spanish_explanation,
            }
        ],
        "difficulty": "A1",
    }
    with patch("app.services.llm._call_anthropic", new=AsyncMock(return_value=mock_raw)):
        result = await LLMService().get_feedback(
            FeedbackRequest(
                sentence="La chat est mignon.",
                target_language="French",
                native_language="Spanish",
            )
        )

    assert result.errors[0].explanation == spanish_explanation


# ── Input validation tests ────────────────────────────────────────────────────


def test_invisible_unicode_stripped_from_sentence():
    """Verifies that invisible Unicode control characters (zero-width space,
    RTL override, etc.) are silently stripped before the sentence reaches the LLM.

    This prevents prompt-injection via non-printable characters that are
    invisible in JSON payloads but visible to the model.
    """
    # U+200B = zero-width space (Cf), U+202E = RTL override (Cf)
    dirty = "Hola\u200b mundo\u202e"
    req = FeedbackRequest(
        sentence=dirty, target_language="Spanish", native_language="English"
    )
    # After stripping, the sentence should contain only the visible characters.
    assert "\u200b" not in req.sentence
    assert "\u202e" not in req.sentence
    assert "Hola" in req.sentence


def test_sentence_exceeding_max_length_rejected():
    """Verifies that a sentence longer than 2000 characters is rejected with
    a Pydantic ValidationError (which FastAPI converts to HTTP 422).

    This guard prevents accidental submission of entire paragraphs or
    adversarial payloads designed to exhaust the model's context window.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FeedbackRequest(
            sentence="x" * 2001,
            target_language="English",
            native_language="English",
        )


# ── Few-shot tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_few_shot_prefix_included_and_merged_correctly_in_anthropic_call():
    """Verifies that the few-shot prefix from prompts.py is prepended and that
    the last message correctly merges the required tool_result with the actual
    user request — the only valid structure for Anthropic tool-use few-shot.

    Anthropic requires every tool_use to be followed by a tool_result in the
    next user message.  The merged final message satisfies this constraint
    while avoiding consecutive user messages.
    """
    from app.services.prompts import ACTIVE_FEW_SHOT_LAST_TOOL_USE_ID, ACTIVE_FEW_SHOT_PREFIX

    captured: list = []

    async def capturing_mock(system: str, messages: list) -> dict:
        captured.extend(messages)
        return _correct_raw("Test sentence.")

    from app.services.cache import _cache, _lock
    with _lock:
        _cache.clear()

    with patch("app.services.llm._call_anthropic", new=capturing_mock):
        await LLMService().get_feedback(_make_request("Test sentence.", target="English"))

    # Prefix messages must appear first.
    assert len(captured) == len(ACTIVE_FEW_SHOT_PREFIX) + 1

    # Last message must be a user turn with a tool_result + text block (merged).
    last = captured[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    types = [block["type"] for block in last["content"]]
    assert "tool_result" in types, "Last user message must contain a tool_result block"
    assert "text" in types, "Last user message must contain the actual request as a text block"

    # The tool_result must reference the last few-shot tool_use ID.
    tool_result_block = next(b for b in last["content"] if b["type"] == "tool_result")
    assert tool_result_block["tool_use_id"] == ACTIVE_FEW_SHOT_LAST_TOOL_USE_ID

    # The text block must contain the actual sentence.
    text_block = next(b for b in last["content"] if b["type"] == "text")
    assert "Test sentence." in text_block["text"]


# ── Concurrency / in-flight dedup tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_in_flight_dedup_fires_only_one_llm_call_for_concurrent_requests():
    """Verifies the in-flight Future map prevents N concurrent identical requests
    from triggering N LLM calls.

    This is the 'thundering herd' protection: if a classroom of 30 students
    submits the same exercise sentence simultaneously, only ONE Anthropic API
    call is made and all 30 receive the result.
    """
    sentence = "Bonjour le monde."
    request = _make_request(sentence, target="French")

    from app.services.cache import _cache, _lock
    from app.services.llm import _in_flight

    # Clear state so prior test runs don't pollute this one.
    with _lock:
        _cache.clear()
    _in_flight.clear()

    call_count = 0
    # Introduce a small delay to ensure concurrent requests actually overlap.
    async def slow_mock(*args, **kwargs) -> dict:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return _correct_raw(sentence)

    with patch("app.services.llm._call_anthropic", new=slow_mock):
        service = LLMService()
        # Fire 5 identical requests simultaneously.
        results = await asyncio.gather(*[service.get_feedback(request) for _ in range(5)])

    assert call_count == 1, (
        f"Expected exactly 1 LLM call for 5 concurrent identical requests, got {call_count}"
    )
    # All 5 responses must be identical.
    assert all(r == results[0] for r in results)


# ── Batch endpoint tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_endpoint_returns_results_for_all_sentences():
    """Verifies that POST /feedback/batch returns one result per sentence
    in the same order as the request list, with correct success counts.
    """
    sentences = [
        "Je mange une pomme.",
        "La chat est mignon.",
    ]

    async def mock_feedback(req: FeedbackRequest) -> FeedbackResponse:
        if "chat" in req.sentence:
            return FeedbackResponse(
                corrected_sentence="Le chat est mignon.",
                is_correct=False,
                errors=[
                    {
                        "original": "La chat",
                        "correction": "Le chat",
                        "error_type": "gender_agreement",
                        "explanation": "Chat is masculine.",
                    }
                ],
                difficulty="A1",
            )
        return FeedbackResponse(
            corrected_sentence=req.sentence,
            is_correct=True,
            errors=[],
            difficulty="A1",
        )

    with patch.object(LLMService, "get_feedback", side_effect=mock_feedback):
        from app.core.dependencies import get_llm_service
        from app.models.schemas import BatchFeedbackRequest
        from app.api.routes import feedback_batch

        result = await feedback_batch(
            BatchFeedbackRequest(
                sentences=sentences,
                target_language="French",
                native_language="English",
            ),
            service=LLMService(),
        )

    assert result.total == 2
    assert result.succeeded == 2
    assert result.failed == 0
    # Order must match the input list.
    assert result.results[0].sentence == sentences[0]
    assert result.results[1].sentence == sentences[1]
    assert result.results[0].result.is_correct is True
    assert result.results[1].result.is_correct is False


@pytest.mark.asyncio
async def test_batch_endpoint_partial_failure_does_not_abort_batch():
    """Verifies that one failing sentence in a batch does not abort the other
    sentences — the response includes both a result and an error entry.

    This behaviour is critical for classroom use cases where a network blip
    or model refusal on one sentence shouldn't fail the whole class's homework.
    """
    call_count = 0

    async def flaky_mock(req: FeedbackRequest) -> FeedbackResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated LLM failure")
        return FeedbackResponse(
            corrected_sentence=req.sentence,
            is_correct=True,
            errors=[],
            difficulty="A1",
        )

    with patch.object(LLMService, "get_feedback", side_effect=flaky_mock):
        from app.models.schemas import BatchFeedbackRequest
        from app.api.routes import feedback_batch

        result = await feedback_batch(
            BatchFeedbackRequest(
                sentences=["Good sentence.", "Bad sentence."],
                target_language="English",
                native_language="English",
            ),
            service=LLMService(),
        )

    assert result.total == 2
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.results[0].error is None
    assert result.results[1].error is not None
    assert "Simulated LLM failure" in result.results[1].error
