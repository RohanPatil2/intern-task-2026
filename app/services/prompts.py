"""Versioned, modular prompt templates for language feedback.

Keeping prompts in their own module means:
  - Prompt changes can be reviewed, tested, and rolled back independently.
  - A/B testing is trivial: add SYSTEM_PROMPT_V2, update ACTIVE_*.
  - Token counts are easy to audit and optimise in one place.

Few-shot strategy (Anthropic tool-use):
  Anthropic's API enforces a strict rule: every tool_use block in an
  assistant message must be immediately followed by a tool_result block in
  the next user message.  To fit two examples without violating this rule
  and without creating consecutive user messages (also forbidden), the
  messages are structured as:

    [0] user:      Example 1 input
    [1] assistant: tool_use  (id=fs_A)
    [2] user:      tool_result(fs_A)  +  Example 2 input   ← MERGED
    [3] assistant: tool_use  (id=fs_B)
    --- runtime ---
    [4] user:      tool_result(fs_B)  +  actual request    ← MERGED at runtime

  The merge at position [4] happens in LLMService.get_feedback() using
  FEW_SHOT_LAST_TOOL_USE_ID so the prefix and runtime code stay decoupled.

  Why turn-based few-shot beats system-prompt examples:
    The model has never "seen" itself call the tool in a text description.
    Showing real tool_use blocks in prior turns grounds the output schema
    more reliably, especially for the is_correct=true / empty-errors case.
"""

from typing import Any

# ── Version 1: system prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT_V1 = """\
You are a supportive language tutor helping learners improve their writing.
Your feedback must be accurate, encouraging, and written for the learner — not a linguist.

For each sentence you receive:
1. Identify every error: grammar, spelling, word choice, punctuation, word order,
   missing/extra words, conjugation, gender/number agreement, tone, or other.
2. Provide the MINIMAL correction that preserves the learner's original meaning
   and voice. Fix only what is wrong — do not paraphrase or "improve" style.
3. Rate CEFR difficulty (A1–C2) based on the sentence's vocabulary and structural
   complexity, NOT on how many errors it contains.
4. Write every explanation in the learner's NATIVE language (given below) so they
   can fully understand the feedback without needing the target language.

Explanation style:
- Be concise (1–2 sentences) but clear.
- Use plain language — avoid jargon like "nominative case" unless the learner's
  level warrants it; prefer "subject form of the word" instead.
- Be encouraging: frame errors as common learning milestones, not failures.
  e.g. "This is a very common mix-up for English speakers" is better than
  "This is wrong".

Critical rules — follow these exactly:
- If the sentence has NO errors: set is_correct=true, errors=[], and
  corrected_sentence must equal the original sentence character-for-character.
- Never flag correct sentences as wrong; never invent errors to seem thorough.
- Handle all writing systems correctly: Latin, Cyrillic, CJK, Arabic,
  Devanagari, Hangul, Hebrew, Thai, and others.\
"""

# ── Version 1: user message template ─────────────────────────────────────────

USER_PROMPT_V1 = (
    "Sentence: {sentence}\n"
    "Target language: {target_language}\n"
    "Learner's native language: {native_language}"
)

# ── Few-shot conversation prefix ──────────────────────────────────────────────
# See module docstring for the structural rationale.
#
# Two examples are enough to anchor the model:
#   Example A — French gender agreement error  → demonstrates error detection,
#               native-language explanation, minimal correction, CEFR rating.
#   Example B — Correct German sentence        → demonstrates is_correct=true,
#               empty errors, corrected_sentence == original (anti-hallucination).

FEW_SHOT_PREFIX: list[dict[str, Any]] = [
    # ── Example A: one gender-agreement error in French ───────────────────────
    {
        "role": "user",
        "content": (
            "Sentence: Je suis allé au magasin avec ma ami.\n"
            "Target language: French\n"
            "Learner's native language: English"
        ),
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "fs_A",
                "name": "language_feedback",
                "input": {
                    "corrected_sentence": "Je suis allé au magasin avec mon ami.",
                    "is_correct": False,
                    "errors": [
                        {
                            "original": "ma ami",
                            "correction": "mon ami",
                            "error_type": "gender_agreement",
                            "explanation": (
                                "'Ami' (friend) is masculine in French. "
                                "Use the masculine possessive 'mon', not the feminine 'ma'."
                            ),
                        }
                    ],
                    "difficulty": "A2",
                },
            }
        ],
    },
    # After a tool_use assistant turn, the next user turn MUST contain a
    # tool_result block.  We merge that required result with Example B's input
    # so the messages list stays valid without adding an extra turn.
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "fs_A",
                "content": "Feedback recorded.",
            },
            {
                "type": "text",
                "text": (
                    "Sentence: Ich lese jeden Tag ein Buch.\n"
                    "Target language: German\n"
                    "Learner's native language: English"
                ),
            },
        ],
    },
    # ── Example B: correct German sentence — no errors ────────────────────────
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "fs_B",
                "name": "language_feedback",
                "input": {
                    "corrected_sentence": "Ich lese jeden Tag ein Buch.",
                    "is_correct": True,
                    "errors": [],
                    "difficulty": "A1",
                },
            }
        ],
    },
    # NOTE: The tool_result for fs_B is NOT included here.
    # It is merged with the actual user request at runtime in
    # LLMService.get_feedback() using FEW_SHOT_LAST_TOOL_USE_ID below.
]

# The ID of the last tool_use in FEW_SHOT_PREFIX — used by LLMService to
# construct the merged tool_result + actual-request user message.
FEW_SHOT_LAST_TOOL_USE_ID = "fs_B"

# ── Active versions — change these to roll out new prompts globally ───────────

ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V1
ACTIVE_USER_PROMPT = USER_PROMPT_V1
ACTIVE_FEW_SHOT_PREFIX = FEW_SHOT_PREFIX
ACTIVE_FEW_SHOT_LAST_TOOL_USE_ID = FEW_SHOT_LAST_TOOL_USE_ID
