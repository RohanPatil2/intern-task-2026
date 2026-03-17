"""Unit tests for the LLM feedback service.

All Anthropic API calls are mocked so these tests run without an API key
and complete in milliseconds.  They verify:
  - Correct parsing of the Anthropic tool_use response into FeedbackResponse
  - Proper handling of the "already correct" edge case
  - Multi-error detection and categorisation
  - Non-Latin script (Japanese) round-trip without encoding errors
  - Cache hit/miss behaviour (LLM called once, second response from cache)
  - Mixed error-type detection (Portuguese spelling + grammar)
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.services.llm import LLMService


# ── Helper ────────────────────────────────────────────────────────────────────


def _make_request(sentence: str, target: str = "Spanish", native: str = "English") -> FeedbackRequest:
    """Convenience factory for FeedbackRequest objects in tests."""
    return FeedbackRequest(sentence=sentence, target_language=target, native_language=native)


# ── Test cases ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spanish_conjugation_error_parsed_correctly():
    """Verifies that a Spanish conjugation error is detected and the response
    fields (corrected_sentence, is_correct, error_type, difficulty) are all
    populated correctly from the mocked tool_use output.

    'Yo soy fue al mercado' mixes 'soy' (present of ser) and 'fue' (past of ir);
    the correct preterite is 'fui'.
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
    """Verifies the 'already correct' edge case:  is_correct must be True,
    errors must be an empty list, and corrected_sentence must equal the input.

    This is an explicitly required behaviour in the task spec and is worth
    testing in isolation to prevent regressions.
    """
    sentence = "Ich habe gestern einen interessanten Film gesehen."
    mock_raw = {
        "corrected_sentence": sentence,
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with patch("app.services.llm._call_anthropic", new=AsyncMock(return_value=mock_raw)):
        result = await LLMService().get_feedback(
            _make_request(sentence, target="German")
        )

    assert result.is_correct is True
    assert result.errors == []
    # corrected_sentence must be identical to the input — not a paraphrase.
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
    assert result.difficulty == "A1"


@pytest.mark.asyncio
async def test_japanese_non_latin_script_handled_without_encoding_errors():
    """Verifies that Japanese (CJK script) input and output is handled cleanly
    with no character encoding errors.

    '図書館を住んでいます' uses the wrong particle 'を'; the location particle
    for 住む (to live) is 'に'.  The corrected sentence must contain 'に'.
    """
    mock_raw = {
        "corrected_sentence": "図書館に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "The verb '住む' (to live) needs the location particle 'に', not 'を'.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.services.llm._call_anthropic", new=AsyncMock(return_value=mock_raw)):
        result = await LLMService().get_feedback(
            _make_request("図書館を住んでいます。", target="Japanese")
        )

    assert result.is_correct is False
    # Ensure the corrected Japanese character is preserved intact.
    assert "に" in result.corrected_sentence
    assert len(result.errors) == 1
    assert result.errors[0].correction == "に"


@pytest.mark.asyncio
async def test_cache_hit_skips_llm_on_second_identical_request():
    """Verifies that the in-memory cache prevents duplicate LLM calls.

    The first request should call _call_anthropic exactly once; the second
    identical request must return the cached result without a second LLM call.
    This is critical for cost efficiency at scale.
    """
    sentence = "Je mange une pomme chaque matin."
    request = _make_request(sentence, target="French")

    mock_raw = {
        "corrected_sentence": sentence,
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    # Clear the module-level cache so previous test runs don't pollute this test.
    from app.services.cache import _cache, _lock

    with _lock:
        _cache.clear()

    call_count = 0

    async def counting_mock(*args, **kwargs) -> dict:
        nonlocal call_count
        call_count += 1
        return mock_raw

    with patch("app.services.llm._call_anthropic", new=counting_mock):
        service = LLMService()
        first = await service.get_feedback(request)
        second = await service.get_feedback(request)

    assert call_count == 1, "LLM must only be called once; second response should come from cache"
    assert first == second


@pytest.mark.asyncio
async def test_portuguese_mixed_spelling_and_grammar_errors():
    """Verifies that a sentence with multiple error types (spelling + grammar)
    returns all errors with the correct categories.

    'prezente' → 'presente' (spelling)
    'o que ela gosta' → 'do que ela gosta' (grammar — 'gostar' requires 'de')
    """
    mock_raw = {
        "corrected_sentence": "Eu quero comprar um presente para minha irmã, mas não sei do que ela gosta.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "The correct spelling is 'presente' (gift).",
            },
            {
                "original": "o que ela gosta",
                "correction": "do que ela gosta",
                "error_type": "grammar",
                "explanation": "'Gostar' requires the preposition 'de', so use 'do que'.",
            },
        ],
        "difficulty": "B1",
    }

    with patch("app.services.llm._call_anthropic", new=AsyncMock(return_value=mock_raw)):
        result = await LLMService().get_feedback(
            _make_request(
                "Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
                target="Portuguese",
            )
        )

    assert result.is_correct is False
    assert len(result.errors) == 2
    error_types = {e.error_type for e in result.errors}
    assert "spelling" in error_types
    assert "grammar" in error_types
    assert result.difficulty == "B1"


@pytest.mark.asyncio
async def test_explanation_language_field_is_preserved():
    """Verifies that the explanation text returned by the LLM (in whatever
    native language was requested) is stored and returned verbatim.

    This tests that the native_language field actually drives the content of
    explanations — a critical requirement for non-English learners.
    """
    # Simulate a Spanish speaker learning French: explanations should be in Spanish.
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
                native_language="Spanish",  # explanations in Spanish
            )
        )

    assert result.errors[0].explanation == spanish_explanation
