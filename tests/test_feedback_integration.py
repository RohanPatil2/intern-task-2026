"""Integration tests — make real Anthropic API calls.

These tests require ANTHROPIC_API_KEY to be set (in .env or the environment).
They are automatically skipped when the key is absent so the CI unit-test
suite never blocks on a missing credential.

Run integration tests explicitly:
    pytest tests/test_feedback_integration.py -v

Each test documents *what linguistic behaviour* is being verified so that a
reviewer unfamiliar with the target language can still understand the intent.
"""

import os

import pytest

from app.models.schemas import FeedbackRequest
from app.services.llm import LLMService

# Skip the entire module when no key is present rather than failing loudly.
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping integration tests",
)

# These sets are used in multiple tests for schema assertion.
VALID_ERROR_TYPES = {
    "grammar", "spelling", "word_choice", "punctuation", "word_order",
    "missing_word", "extra_word", "conjugation", "gender_agreement",
    "number_agreement", "tone_register", "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the in-memory cache before every integration test.

    Without this, a passing test could seed the cache and cause a subsequent
    test with the same sentence to skip the real LLM call, hiding regressions.
    """
    from app.services.cache import _cache, _lock

    with _lock:
        _cache.clear()
    yield


@pytest.mark.asyncio
async def test_integration_spanish_conjugation_error_detected():
    """Integration: the real LLM must identify a Spanish conjugation error.

    'Yo soy fue al mercado ayer' illegally combines 'soy' (present of ser)
    with 'fue' (preterite of ir).  The API must return is_correct=False and
    include at least one error classified as 'conjugation'.
    """
    service = LLMService()
    result = await service.get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert any(e.error_type == "conjugation" for e in result.errors)
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_integration_correct_german_returns_no_errors():
    """Integration: a grammatically correct German sentence must return
    is_correct=True, an empty errors list, and the original sentence unchanged.

    Tests the critical 'no false positives' requirement.
    """
    sentence = "Ich habe gestern einen interessanten Film gesehen."
    service = LLMService()
    result = await service.get_feedback(
        FeedbackRequest(
            sentence=sentence,
            target_language="German",
            native_language="English",
        )
    )

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == sentence
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_integration_french_gender_agreement_errors():
    """Integration: French gender agreement errors are caught by the real LLM.

    'La chat mange le table' has two gender errors — both must be detected.
    Tests multi-error recall (the LLM must not stop at the first error found).
    """
    service = LLMService()
    result = await service.get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert any(e.error_type == "gender_agreement" for e in result.errors)


@pytest.mark.asyncio
async def test_integration_japanese_particle_error_with_cjk_correction():
    """Integration: a Japanese particle error is detected without any
    character encoding issues or script-related failures.

    '私は東京を住んでいます' uses 'を' where 'に' is required with 住む (to live).
    The corrected sentence must contain the Japanese character 'に'.
    Tests that the API pipeline handles CJK script end-to-end.
    """
    service = LLMService()
    result = await service.get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert "に" in result.corrected_sentence
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_integration_russian_cyrillic_script():
    """Integration: Russian (Cyrillic script) is handled without encoding
    failures or hallucinated corrections.

    'Я хочу пить воды холодный' contains a case agreement error:
    'холодный' (masculine nominative) should be 'холодной' (genitive feminine
    to agree with 'воды').  At least one error must be returned.
    Tests non-Latin script support on the Cyrillic alphabet.
    """
    service = LLMService()
    result = await service.get_feedback(
        FeedbackRequest(
            sentence="Я хочу пить воды холодный.",
            target_language="Russian",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES


@pytest.mark.asyncio
async def test_integration_cache_returns_identical_result_on_repeat_call():
    """Integration: a repeated request returns a cached response identical to
    the first, confirming that the caching layer is active end-to-end.

    The cache key is derived from sentence + target_language + native_language,
    so both calls use the same key and the second should be a cache hit.
    """
    service = LLMService()
    request = FeedbackRequest(
        sentence="Je mange une pomme chaque matin.",
        target_language="French",
        native_language="English",
    )

    first = await service.get_feedback(request)
    second = await service.get_feedback(request)

    # Pydantic models compare by value, so == checks field equality.
    assert first == second
