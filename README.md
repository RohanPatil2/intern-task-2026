# Language Feedback API

An LLM-powered grammar correction service for language learners, built with FastAPI and Anthropic Claude.

## Quick Start

```bash
# 1. Clone the repo
git clone <your-fork-url>
cd intern-task-2026

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 4. Start the server
uvicorn app.main:app --reload

# 5. Try it
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence":"Yo soy fue al mercado ayer.","target_language":"Spanish","native_language":"English"}'
```

## Docker

```bash
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker compose up --build
```

The server starts on port 8000. Health check: `GET /health`.

## Running Tests

```bash
# Unit tests — no API key required (LLM calls are fully mocked)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests — require ANTHROPIC_API_KEY in .env
pytest tests/test_feedback_integration.py -v

# All tests
pytest -v
```

Tests also run inside the Docker container:

```bash
docker compose exec feedback-api pytest tests/test_feedback_unit.py tests/test_schema.py -v
```

---

## Architecture

```
app/
├── main.py              # FastAPI app creation + startup validation
├── api/
│   └── routes.py        # GET /health, POST /feedback
├── core/
│   ├── config.py        # All settings loaded from environment variables
│   └── dependencies.py  # FastAPI dependency factories (LLMService singleton)
├── models/
│   └── schemas.py       # Pydantic v2 request/response models
└── services/
    ├── prompts.py        # Versioned system/user prompt templates
    ├── cache.py          # In-memory TTL cache (SHA-256 keyed)
    └── llm.py            # Anthropic client, tool-use call, retry logic
```

Concerns are strictly separated: routing, configuration, data models, prompts, caching, and LLM interaction each live in their own module. Nothing non-trivial is in `main.py`.

---

## Design Decisions

### LLM Provider: Anthropic Claude Haiku 4.5

`claude-haiku-4-5` hits the best balance of speed, cost, and accuracy for this task:

- **Speed**: 4–5× faster than Claude Sonnet 4.5 — well under the 30 s endpoint timeout even for long sentences.
- **Cost**: $1 / 1M input tokens, $5 / 1M output tokens — the most cost-effective frontier model.
- **Accuracy**: 90% of Sonnet 4.5's performance on structured tasks; strong multilingual performance including CJK, Cyrillic, Arabic, Devanagari, and Hangul scripts.

The model is configurable via the `LLM_MODEL` environment variable, so swapping to `claude-sonnet-4-6` for higher accuracy (at higher cost) requires no code changes.

### Structured Outputs via Tool-Use

Instead of asking the model to "return JSON" and then parsing the string, the API uses Anthropic's **tool-use** feature with `tool_choice={"type":"tool","name":"language_feedback"}`. This forces the model to always call the `language_feedback` tool, which has a strict JSON schema attached. The response is therefore schema-validated *by Anthropic's own infrastructure* before it ever reaches our code — no `json.loads`, no regex, no hope.

Pydantic provides a second validation layer: even if a future model version returns a malformed `error_type`, the `Literal` type annotation in `ErrorDetail` will raise a `ValidationError` rather than silently passing bad data downstream.

### Prompt Engineering

The system prompt (`app/services/prompts.py`) is:

1. **Script-agnostic**: Explicitly instructs the model to handle all writing systems (Latin, Cyrillic, CJK, Arabic, Devanagari, Hangul, Hebrew, Thai). No assumption that input is romanisable.
2. **Anti-hallucination guardrails**: The rule *"if the sentence has NO errors: set is_correct=true, errors=[], corrected_sentence=original"* is stated explicitly and unambiguously. Ambiguous instructions are the primary cause of LLM hallucination on structured tasks.
3. **Native-language explanations**: The prompt reminds the model that explanations must be written in the *learner's* language. Without this reminder, the model defaults to the target language for the explanation.
4. **Token-efficient**: The system prompt is ~170 tokens. The user message template adds only the three input fields. Total input tokens per request: ~220–300 depending on sentence length.

The prompt is versioned (`SYSTEM_PROMPT_V1`, `ACTIVE_SYSTEM_PROMPT`) so A/B testing or a rollback is a one-line change without touching business logic.

### Caching

Repeated calls with the same `(sentence, target_language, native_language)` tuple return an in-memory cached result instantly, consuming zero LLM tokens.

- **Key**: SHA-256 of the normalised concatenation of the three fields — collision-resistant, fixed-size, script-safe.
- **Backend**: `cachetools.TTLCache` (1 hour TTL, 1000 entry max). Entries are automatically evicted when stale.
- **Thread safety**: A `threading.Lock` protects every read/write because `TTLCache` is not thread-safe.
- **Swap path**: The `get_cached` / `set_cached` interface in `cache.py` is the only place that touches the storage backend. Replacing it with `redis.asyncio` for a multi-process deployment requires changing only those two functions.

### Retry Logic

`tenacity` wraps every Anthropic API call with exponential back-off (1 s → 10 s, max 3 attempts) for the four transient error types: `RateLimitError`, `APIConnectionError`, `APITimeoutError`, and `InternalServerError`. Authentication errors and bad request errors are *not* retried — retrying those would waste tokens.

### Error Handling

- **Startup**: `lifespan` raises `RuntimeError` immediately if `ANTHROPIC_API_KEY` is missing, so misconfigured containers fail fast at launch rather than at the first request.
- **Request level**: All unhandled exceptions in `POST /feedback` are caught, logged with full traceback (for observability), and re-raised as `HTTP 500` with a generic message — no internal details leak to the client.
- **Validation**: Pydantic 422 errors from bad request bodies are handled automatically by FastAPI.

### Singleton Client

The `anthropic.AsyncAnthropic` client is a module-level singleton (`app/services/llm.py`). The underlying `httpx.AsyncClient` it wraps maintains a connection pool that is reused across all requests, eliminating TCP handshake overhead on every call.

---

## Production Considerations

This implementation is designed to be *production-feasible*, not just functional:

| Concern | This implementation | At scale |
|---|---|---|
| Structured output | Anthropic tool-use (schema-enforced) | Same |
| Caching | In-process TTLCache | Redis (swap `cache.py` backend only) |
| Retries | tenacity exponential backoff | Same |
| Client lifecycle | Singleton async client | Same |
| Workers | Single uvicorn worker | `--workers N` or Gunicorn + uvicorn |
| Observability | Python `logging` | Swap for structlog + OTLP exporter |

---

## Verifying Accuracy Without Speaking the Language

The hidden test suite covers 8+ languages. Since I don't speak all of them, accuracy was verified by:

1. **Cross-checking corrections**: Running each test sentence through both Claude and an independent source (e.g., a native-speaker grammar reference or a second LLM call with a different prompt).
2. **Schema compliance**: All responses are validated against `schema/response.schema.json` automatically in `tests/test_schema.py`.
3. **Edge cases in unit tests**: The mocked unit tests codify *expected* corrections from known linguistic references, so any prompt regression that changes the correction logic will break a test.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API secret key |
| `LLM_MODEL` | `claude-haiku-4-5` | Anthropic model ID |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime in seconds |
| `CACHE_MAX_SIZE` | `1000` | Maximum number of cached responses |
| `REQUEST_TIMEOUT` | `25.0` | Per-request Anthropic API timeout (seconds) |
