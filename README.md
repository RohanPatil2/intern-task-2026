<div align="center">

# 🌍 Language Feedback API

### *Instant, structured grammar correction for language learners — powered by LLMs*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude%20Haiku%204.5-CC785C?style=for-the-badge&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Tests](https://img.shields.io/badge/Tests-30%20passing-4CAF50?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

</div>

---

## 📖 Overview

**Language Feedback API** is a production-grade REST API that acts like a private language tutor available 24/7. You send it a sentence written by a language learner and it sends back:

- ✅ A **corrected version** of the sentence (minimal edits only — preserving the learner's voice)
- 🔍 A **detailed list of every error** — what it was, what it should be, why it was wrong
- 📊 A **CEFR difficulty rating** (A1 → C2) based on the sentence's complexity
- 🌐 **Explanations written in the learner's own native language**

It works across **any language and any writing system** — Latin, Cyrillic, Japanese (CJK), Arabic, Devanagari, Hangul, and more.

---

## ✨ Features

- **🔒 Schema-guaranteed responses** — Uses Anthropic's tool-use API to *force* structured JSON output. No raw string parsing, no guesswork.
- **⚡ Two-layer deduplication** — Persistent TTL cache + asyncio in-flight dedup means a popular sentence is only ever sent to the LLM *once*, no matter how many students submit it simultaneously.
- **📦 Batch analysis** — `POST /feedback/batch` analyses up to 10 sentences concurrently in a single request. Perfect for reviewing a whole class's homework.
- **🔁 Automatic retry logic** — Transient API errors (rate limits, connection drops) are retried with exponential back-off via `tenacity` — invisible to the caller.
- **🧠 Turn-based few-shot prompting** — Two carefully chosen example conversations are prepended to every LLM call, grounding the model's output format and dramatically reducing hallucination.
- **🛡️ Input hardening** — Invisible Unicode control characters (zero-width joiners, RTL overrides) are stripped before processing. Sentences over 2 000 characters are rejected with a clear error.
- **📡 Observability** — `GET /stats` returns live cache hit rate, miss count, and number of in-flight LLM calls. Every response carries `X-Request-ID` and `X-Response-Time-Ms` headers.
- **🐳 One-command Docker deployment** — `docker compose up --build` and you're live on port 8000.

---

## 🏗️ How It Works — Under the Hood

Here's the full journey of a single request, explained simply:

### Step 1 — Request arrives and is validated
A `POST /feedback` request lands on FastAPI. Before anything else, **Pydantic v2** validates the request body against a strict schema. If the sentence is blank, too long, or contains invisible Unicode tricks, it's rejected immediately with a `422` error — the LLM never even sees it.

### Step 2 — Cache lookup (Layer 1)
A **SHA-256 hash** of the sentence + target language + native language is computed. This hash is checked against an in-memory **TTL cache** (1-hour lifetime, 1 000 entries max). If this exact sentence has been asked before, the stored response is returned *instantly* — zero API tokens consumed.

### Step 3 — In-flight deduplication (Layer 2)
If the sentence *isn't* in the cache, the code checks a second map: are there already concurrent requests for this same sentence right now? If yes, this request **shares the Future** of the first request — it waits for it to finish and receives the same result. This prevents 30 students submitting the same exercise from triggering 30 separate LLM calls.

```
Request A ──▶ cache miss ──▶ starts LLM call ──┐
Request B ──▶ cache miss ──▶ finds in-flight ──┤─▶ both get same result
Request C ──▶ cache miss ──▶ finds in-flight ──┘
```

### Step 4 — Few-shot message construction
The system builds the messages list for Anthropic:
1. **System prompt** — concise instructions: identify errors, correct minimally, use CEFR ratings, write in the native language, never hallucinate errors on correct sentences.
2. **Few-shot turn A** — a real example of a French gender-agreement error being caught correctly (shows the model what a good error detection looks like).
3. **Few-shot turn B** — a correct German sentence being returned unchanged (teaches the model *not* to invent errors — the most common LLM mistake on this task).
4. **The actual user request**.

```
System prompt
  └─▶ Example A: French sentence WITH error  (tool_use + tool_result)
  └─▶ Example B: German CORRECT sentence     (tool_use + tool_result)
  └─▶ Actual learner sentence
```

### Step 5 — LLM call with guaranteed structure
The request is sent to **Claude Haiku 4.5** using Anthropic's **tool-use** feature with `tool_choice={"type":"tool"}`. This forces the model to *always* call our `language_feedback` tool — it cannot respond with plain text. Anthropic validates the output against the tool's JSON schema on their end before returning it. Our Pydantic model validates it again on ours. Two independent schema checks.

### Step 6 — Error type normalisation
If the model ever returns an `error_type` outside the 12 allowed values (e.g. `"case"` instead of `"grammar"`), a normaliser silently remaps it to `"other"` rather than crashing. The learner still gets their feedback.

### Step 7 — Response cached and returned
The validated response is stored in the TTL cache (for future requests) and the asyncio Future is resolved (for any in-flight duplicates). The response is returned to the caller with `X-Request-ID` and `X-Response-Time-Ms` headers attached by middleware.

---

## 📁 Project Structure

```
intern-task-2026/
│
├── app/
│   ├── main.py                  # FastAPI app, CORS, correlation-ID middleware
│   │
│   ├── api/
│   │   └── routes.py            # POST /feedback, POST /feedback/batch, GET /health, GET /stats
│   │
│   ├── core/
│   │   ├── config.py            # All settings loaded from environment variables
│   │   └── dependencies.py      # FastAPI dependency injection (LLMService singleton)
│   │
│   ├── models/
│   │   └── schemas.py           # Pydantic v2 models: request, response, batch, error detail
│   │
│   └── services/
│       ├── prompts.py           # Versioned system prompt + few-shot message templates
│       ├── cache.py             # SHA-256 keyed TTL cache with hit/miss telemetry
│       └── llm.py               # Anthropic client, in-flight dedup, retry, response parsing
│
├── tests/
│   ├── test_feedback_unit.py    # 13 unit tests — fully mocked, no API key needed
│   ├── test_feedback_integration.py  # 8 integration tests — real Anthropic API calls
│   └── test_schema.py           # 9 JSON Schema validation tests
│
├── schema/
│   ├── request.schema.json      # JSON Schema for request validation
│   └── response.schema.json     # JSON Schema for response validation
│
├── examples/
│   └── sample_inputs.json       # 5 example request/response pairs
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) (free trial credits available)
- Docker & Docker Compose (optional, for containerised deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone <your-fork-url>
cd intern-task-2026

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=your-key-here

# 5. Start the server
uvicorn app.main:app --reload
```

The API is now live at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

### Docker (recommended for production)

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

docker compose up --build
```

That's it. The server starts on port 8000 with a built-in health check.

---

## 🔌 API Reference

### `POST /feedback` — Single sentence analysis

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "Yo soy fue al mercado ayer.",
    "target_language": "Spanish",
    "native_language": "English"
  }'
```

**Response:**
```json
{
  "corrected_sentence": "Yo fui al mercado ayer.",
  "is_correct": false,
  "errors": [
    {
      "original": "soy fue",
      "correction": "fui",
      "error_type": "conjugation",
      "explanation": "You mixed two verb forms. 'Soy' is present tense of 'ser', and 'fue' is past tense of 'ir'. Use 'fui' (I went)."
    }
  ],
  "difficulty": "A2"
}
```

---

### `POST /feedback/batch` — Multiple sentences at once

```bash
curl -X POST http://localhost:8000/feedback/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sentences": [
      "Je mange une pomme.",
      "La chat est sur le table."
    ],
    "target_language": "French",
    "native_language": "English"
  }'
```

**Response:**
```json
{
  "results": [
    { "sentence": "Je mange une pomme.", "result": { "is_correct": true, ... }, "error": null },
    { "sentence": "La chat est sur le table.", "result": { "is_correct": false, ... }, "error": null }
  ],
  "total": 2,
  "succeeded": 2,
  "failed": 0
}
```

---

### `GET /health` — Liveness probe
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `GET /stats` — Cache telemetry
```bash
curl http://localhost:8000/stats
# {"cache_size": 12, "hits": 45, "misses": 12, "hit_rate": 0.789, "in_flight_requests": 0}
```

---

## 🧪 Running Tests

```bash
# Unit tests — no API key needed (all LLM calls are mocked)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests — requires ANTHROPIC_API_KEY in .env
pytest tests/test_feedback_integration.py -v

# Full suite
pytest tests/ -v
```

Tests also run inside Docker (as required by the scoring rubric):
```bash
docker compose exec feedback-api pytest tests/test_feedback_unit.py tests/test_schema.py -v
```

**Current status: 30/30 passing ✅**

| Suite | Tests | Covers |
|---|---|---|
| Unit | 13 | Parsing, caching, dedup, batch, input validation, few-shot structure |
| Integration | 8 | Spanish, German, French, Japanese, Russian, batch, cache, stats |
| Schema | 9 | JSON Schema compliance for all request/response shapes |

---

## ⚙️ Configuration

All settings are loaded from environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `LLM_MODEL` | `claude-haiku-4-5` | Anthropic model ID — swap to `claude-sonnet-4-6` for higher accuracy |
| `CACHE_TTL_SECONDS` | `3600` | How long a cached response stays valid (seconds) |
| `CACHE_MAX_SIZE` | `1000` | Maximum number of responses to hold in memory |
| `REQUEST_TIMEOUT` | `25.0` | Per-request LLM timeout in seconds (leaves headroom inside the 30 s limit) |

---

## 🎯 Design Decisions

### Why Anthropic over OpenAI?
The `.env` template ships with `ANTHROPIC_API_KEY`. Claude Haiku 4.5 is the fastest current frontier model (4–5× faster than Sonnet) at the lowest cost ($1/M input tokens), making it the right default for a latency-sensitive classroom tool.

### Why tool-use instead of `json_object` mode?
`response_format={"type":"json_object"}` asks the model to *try* to return valid JSON. Anthropic's tool-use with `tool_choice={"type":"tool"}` *forces* it — the model cannot respond with plain text, and Anthropic validates the output against the tool schema server-side before returning it. Two independent schema checks before data reaches the application.

### Why two-layer deduplication?
A single in-memory cache is not enough for a real classroom scenario. If a teacher asks 30 students to write the same sentence and they all submit it simultaneously — before any of them has been cached — 30 API calls would fire. The asyncio `Future` map catches this: the second through thirtieth requests see the first one in-flight and wait for it instead of starting their own.

### Why few-shot in the messages, not the system prompt?
Text examples in the system prompt describe what a good response looks like. Turn-based examples in the messages array *show* the model what a good tool_use call looks like — a meaningfully stronger signal for structured-output tasks. The examples also specifically cover the "correct sentence → no errors" case, which is the most common source of hallucination on this task.

### Why in-memory cache instead of Redis?
This keeps the deployment a single container with no external dependencies. The `get_cached` / `set_cached` interface in `cache.py` is the only place that touches the storage backend — swapping to `redis.asyncio` for a multi-process deployment requires changing only those two functions.

---

## 🗺️ Production Scaling Path

| Concern | Current | At scale |
|---|---|---|
| Cache backend | `cachetools.TTLCache` (in-process) | `redis.asyncio` — change `cache.py` only |
| Workers | Single uvicorn worker | `uvicorn --workers N` or Gunicorn + uvicorn |
| Observability | Python `logging` + `/stats` endpoint | Swap for structlog + OpenTelemetry exporter |
| Model accuracy | Claude Haiku 4.5 | Set `LLM_MODEL=claude-sonnet-4-6` |
| Rate limiting | None | Add `slowapi` middleware in `main.py` |

---

## 📝 Allowed Error Types

`grammar` · `spelling` · `word_choice` · `punctuation` · `word_order` · `missing_word` · `extra_word` · `conjugation` · `gender_agreement` · `number_agreement` · `tone_register` · `other`

## 📊 CEFR Difficulty Levels

`A1` (beginner) → `A2` → `B1` → `B2` → `C1` → `C2` (mastery)

---

<div align="center">

Built for the **Pangea Chat** Gen AI Intern Task · March 2026

</div>
