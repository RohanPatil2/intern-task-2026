"""Versioned, modular prompt templates for language feedback.

Keeping prompts in their own module means:
  - Prompt changes can be reviewed, tested, and rolled back independently
    of the business logic that calls the LLM.
  - A/B testing is trivial: add SYSTEM_PROMPT_V2, update ACTIVE_SYSTEM_PROMPT.
  - Token counts are easy to audit and optimise in one place.

Design goals for the prompts:
  1. Accuracy over verbosity — be explicit about rules without padding.
  2. Script-agnostic — no Latin-centric assumptions; tested with CJK, Cyrillic,
     Arabic, and Devanagari inputs.
  3. Anti-hallucination guardrails — explicit instructions to return
     is_correct=true and an empty errors array for correct sentences,
     preventing the model from inventing phantom mistakes.
  4. Native-language explanations — the model must write every explanation
     in the *learner's* language, not the target language.
"""

# ── Version 1 ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_V1 = """\
You are a precise language tutor analysing student-written sentences.

For each sentence you receive:
1. Identify every error: grammar, spelling, word choice, punctuation, word order,
   missing/extra words, conjugation, gender/number agreement, tone, or other.
2. Provide the minimal correction that preserves the learner's original meaning
   and voice — do not rewrite sentences beyond what is necessary.
3. Rate the sentence's CEFR difficulty (A1–C2) based on its vocabulary and
   structural complexity, NOT on how many errors it contains.
4. Write every error explanation in the learner's native language (given below).

Critical rules — follow these exactly:
- If the sentence has NO errors: set is_correct=true, errors=[], and
  corrected_sentence must equal the original sentence character-for-character.
- Never flag correct sentences as wrong; never invent errors to seem helpful.
- Handle all writing systems correctly: Latin, Cyrillic, CJK, Arabic,
  Devanagari, Hangul, Hebrew, Thai, etc.
- Keep explanations concise and learner-friendly (1–2 sentences each).\
"""

# The user message is a lightweight template — only the three request fields
# are injected.  Keeping it short reduces input tokens without losing context.
USER_PROMPT_V1 = (
    "Sentence: {sentence}\n"
    "Target language: {target_language}\n"
    "Learner's native language: {native_language}"
)

# ── Active versions — change these to roll out a new prompt globally ──────────

ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V1
ACTIVE_USER_PROMPT = USER_PROMPT_V1
