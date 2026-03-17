"""Pydantic v2 request and response models.

All field types and constraints mirror schema/response.schema.json exactly,
so Pydantic acts as a second validation gate on top of JSON Schema — any
LLM output that slips past tool_use schema enforcement is caught here.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Centralise the allowed enum values so they're shared between the Pydantic
# model, the Anthropic tool definition, and any future OpenAPI documentation.
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
        sentence: The learner's sentence to be analysed.
        target_language: The language the learner is practising.
        native_language: The learner's first language; explanations are written in this language.
    """

    sentence: str = Field(min_length=1, description="The learner's sentence in the target language")
    target_language: str = Field(min_length=2, description="Language being studied")
    native_language: str = Field(
        min_length=2,
        description="Learner's first language — explanations will be written in this language",
    )


class FeedbackResponse(BaseModel):
    """Structured correction feedback returned by POST /feedback.

    Attributes:
        corrected_sentence: Minimal correction that preserves the learner's voice.
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
