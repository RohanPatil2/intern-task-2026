"""Pydantic v2 request and response models with strict validation.

All field types and constraints mirror schema/response.schema.json exactly,
so Pydantic acts as a second validation gate on top of Anthropic's tool-use
schema enforcement.

Input hardening:
  FeedbackRequest strips invisible Unicode control characters (category Cc/Cf)
  before processing to prevent prompt injection via zero-width joiners, RTL
  overrides, or other formatting tricks that could confuse the LLM without
  being visible in the raw JSON.
"""

import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

# Centralised enum values — shared by Pydantic models, the Anthropic tool
# definition, and _normalise_error() in llm.py.
ErrorType = Literal[
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
]

CefrLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


class ErrorDetail(BaseModel):
    """A single identified error within the learner's sentence.

    Attributes:
        original: The erroneous word or phrase copied verbatim from the input.
        correction: The corrected replacement (minimal change).
        error_type: One of the 12 allowed error categories.
        explanation: Learner-friendly explanation written in the native language.
    """

    original: str = Field(description="Erroneous word or phrase from the original sentence")
    correction: str = Field(description="Corrected replacement (minimal change)")
    error_type: ErrorType = Field(description="Error category")
    explanation: str = Field(description="Friendly explanation in the learner's native language")


class FeedbackRequest(BaseModel):
    """Incoming POST /feedback request body.

    Attributes:
        sentence: The learner's sentence to be analysed (max 2000 chars).
        target_language: The language the learner is practising.
        native_language: The learner's first language; explanations are written in this language.
    """

    sentence: str = Field(
        min_length=1,
        max_length=2000,
        description="The learner's sentence in the target language (max 2000 characters)",
    )
    target_language: str = Field(min_length=2, description="Language being studied")
    native_language: str = Field(
        min_length=2,
        description="Learner's first language — explanations will be written in this language",
    )

    @field_validator("sentence", mode="before")
    @classmethod
    def strip_invisible_unicode(cls, v: str) -> str:
        """Remove invisible Unicode control/format characters from the sentence.

        Characters in categories Cc (control) and Cf (format) — such as
        zero-width joiners, RTL/LTR overrides, and soft hyphens — are invisible
        in JSON payloads but can confuse the LLM or be used for prompt injection.

        Args:
            v: Raw sentence string from the request.

        Returns:
            The sentence with all Cc/Cf characters removed.

        Raises:
            ValueError: If the sentence contains only invisible characters.
        """
        cleaned = "".join(
            c for c in v if unicodedata.category(c) not in {"Cc", "Cf"}
        )
        if not cleaned.strip():
            raise ValueError(
                "sentence must contain at least one visible character after stripping control characters"
            )
        return cleaned


class FeedbackResponse(BaseModel):
    """Structured correction feedback returned by POST /feedback.

    Attributes:
        corrected_sentence: Minimal correction preserving the learner's voice.
            Identical to the input when is_correct is True.
        is_correct: True only when the sentence has zero errors.
        errors: Ordered list of identified errors; empty when is_correct is True.
        difficulty: CEFR level of the *input* sentence's linguistic complexity.
    """

    corrected_sentence: str = Field(
        description="Grammatically corrected sentence (identical to input when no errors)"
    )
    is_correct: bool = Field(description="True if the original sentence had no errors")
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="Empty list when the sentence is correct",
    )
    difficulty: CefrLevel = Field(description="CEFR complexity rating of the input sentence")


# ── Batch endpoint models ─────────────────────────────────────────────────────


class BatchFeedbackRequest(BaseModel):
    """Request body for POST /feedback/batch.

    Attributes:
        sentences: 1–10 sentences to analyse in the same language pair.
            All requests share the same target and native language.
        target_language: The language all sentences are written in.
        native_language: The learner's first language for explanations.
    """

    sentences: list[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(
        min_length=1,
        max_length=10,
        description="1 to 10 sentences to analyse (same language pair for all)",
    )
    target_language: str = Field(min_length=2, description="Language being studied")
    native_language: str = Field(min_length=2, description="Learner's native language")


class BatchFeedbackItem(BaseModel):
    """A single result within a batch response.

    Exactly one of ``result`` or ``error`` will be populated.

    Attributes:
        sentence: The original sentence that was analysed.
        result: Populated on success; None on failure.
        error: Human-readable error message on failure; None on success.
    """

    sentence: str
    result: FeedbackResponse | None = None
    error: str | None = None


class BatchFeedbackResponse(BaseModel):
    """Response body for POST /feedback/batch.

    Attributes:
        results: Per-sentence feedback items in the same order as the request.
        total: Total number of sentences submitted.
        succeeded: Number of sentences successfully analysed.
        failed: Number of sentences that encountered an error.
    """

    results: list[BatchFeedbackItem]
    total: int
    succeeded: int
    failed: int
