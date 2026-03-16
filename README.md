# Pangea Chat: Gen AI Intern Task (Summer 2026)

## ⚠️ Before You Begin: Eligibility Requirements

**Do not invest time in this task unless you meet ALL of the following criteria:**

- [ ] You are authorized to work in the United States (or will be by June 2026)
- [ ] You are available for a **10-week on-site internship in Richmond, Virginia** (approximately June–August 2026). Our team works in person and we've found that interns learn faster with same-room collaboration. This is not a remote position.
- [ ] You are proficient in written and spoken English
- [ ] You are currently enrolled in (or have graduated within the past 12 months from) an undergraduate or graduate program, or have equivalent experience

**Preference will be given to candidates interested in full-time employment in Richmond, Virginia after the internship.** The full-time role (pending continued NSF funding) is budgeted at **$70,000–$90,000/year**, which goes further in Richmond than in most tech hubs.

By submitting your solution, you confirm that you meet these requirements. See [RULES.md](RULES.md) for full terms.

---

## About Pangea Chat

[Pangea Chat](https://pangea.chat) is an AI-powered language learning app funded by the U.S. National Science Foundation. Students learn languages by chatting with peers, assisted by AI feedback tools: grammar checking, interactive translation, pronunciation help, and conversation activities.

This task is modeled after real work you'd do as an intern on the team.

## Why This Process

We received over **2,200 applications** for up to **3 intern positions**. Resume screening alone can't give 2,200 people a fair shot. It inevitably favors school names and network connections over actual ability.

So we built this task to let your skills speak for themselves. Everyone gets the same starter code, the same rubric, and the same week. Whether you're at a top-10 CS program or a community college, your submission is scored the same way.

Is automated scoring perfect? No. But it's more equitable than a human skimming 2,200 resumes, and the top submissions all get human review.

## The Task

Build an **LLM-powered language feedback API** that analyzes learner-written sentences and returns structured correction feedback.

Given a sentence in a target language and the learner's native language, your API must return:

1. A **corrected sentence** (minimal edits that preserve the learner's voice)
2. A list of **errors**, each with the original text, correction, error category, and a learner-friendly explanation written in the native language
3. Whether the sentence **is correct** (boolean)
4. A **CEFR difficulty rating** (A1–C2) based on sentence complexity

### Endpoint

```
POST /feedback
```

**Request body** (see [schema/request.schema.json](schema/request.schema.json)):

```json
{
  "sentence": "Yo soy fue al mercado ayer.",
  "target_language": "Spanish",
  "native_language": "English"
}
```

**Response body** (see [schema/response.schema.json](schema/response.schema.json)):

```json
{
  "corrected_sentence": "Yo fui al mercado ayer.",
  "is_correct": false,
  "errors": [
    {
      "original": "soy fue",
      "correction": "fui",
      "error_type": "conjugation",
      "explanation": "You mixed two verb forms. 'Soy' is present tense of 'ser' (to be), and 'fue' is past tense of 'ir' (to go). You only need 'fui' (I went)."
    }
  ],
  "difficulty": "A2"
}
```

### Requirements

1. **Working HTTP endpoint:** must respond to `POST /feedback` with JSON conforming to the response schema
2. **Health check:** `GET /health` must return a 200 response
3. **Prompt engineering:** design a system prompt that reliably produces accurate, well-structured feedback
4. **Test suite:** at least 5 test cases covering different languages, error types, and edge cases (correct sentences, multiple errors, non-Latin scripts)
5. **README:** replace this README explaining your design decisions, prompt strategy, and how to run your solution
6. **Docker support:** `docker compose up` must start your server on port 8000. Your Docker Compose service must be named `feedback-api` (as in the starter repo). Your test dependencies should be installed in the Docker image so tests can run inside the container.
7. **Response time:** each request to `/feedback` must return within **30 seconds**. Requests that exceed this timeout will be treated as failures during scoring.

### Allowed error types

`grammar`, `spelling`, `word_choice`, `punctuation`, `word_order`, `missing_word`, `extra_word`, `conjugation`, `gender_agreement`, `number_agreement`, `tone_register`, `other`

### Allowed difficulty levels

`A1`, `A2`, `B1`, `B2`, `C1`, `C2` (CEFR scale)

## Getting Started

This repo contains a **complete, working submission** out of the box: the API endpoint, Dockerfile, Docker Compose config, JSON schemas, and a basic test suite all work as-is. You do not need to build any infrastructure from scratch. The boilerplate is done so you can focus your time on what matters: **your prompt, your accuracy, your tests, and your code quality.**

Study the sample, then replace or rewrite it with your own approach. The sample is intentionally basic (a straightforward prompt with no caching, no retry logic, and minimal error handling), so treat it as a starting point rather than a ceiling.

### Run locally

```bash
# 1. Clone and enter the repo
git clone https://github.com/pangeachat/intern-task-2026.git
cd intern-task-2026

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your OpenAI or Anthropic API key

# 5. Start the server
uvicorn app.main:app --reload

# 6. Test it
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

### Run with Docker

```bash
cp .env.example .env
# Edit .env with your API key
docker compose up --build
```

### Run tests

```bash
# Unit tests (no API key needed)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires API key in .env)
pytest tests/test_feedback_integration.py -v
```

## How to Submit

1. **Fork** this repository
2. Replace the sample implementation with your own
3. Make sure `docker compose up` works and passes the health check
4. Push to your fork
5. **Fill out the [submission form](https://forms.gle/Nb2aqYvtY12efX2bA)** with your name, email, and the link to your fork

**Use the same email you applied with on Handshake.** This is how we match your submission to your application.

**Deadline: Sunday, March 22, 2026 at 11:59 PM Eastern Time.** Late submissions will not be evaluated.

## How Submissions Are Evaluated

| Criterion                  | Weight | Method      | Description                                                                                    |
| -------------------------- | ------ | ----------- | ---------------------------------------------------------------------------------------------- |
| **Runs successfully**      | Gate   | Automated   | Server starts via Docker and responds to health check. **Fail = disqualified.**                |
| **Schema compliance**      | Gate   | Automated   | ≥90% of responses match JSON schema. **Below threshold = disqualified.**                       |
| **Response time**          | Gate   | Automated   | Each `/feedback` request must return within 30 seconds. **Timeouts = failures.**               |
| **Accuracy**               | 25%    | Absolute    | Corrections are linguistically accurate across a hidden test suite spanning multiple languages |
| **Production feasibility** | 25%    | Comparative | Could this run at scale? Model choice, token efficiency, caching, cost reduction strategy      |
| **Test quality**           | 25%    | Comparative | Your tests are meaningful, cover edge cases, and test real behavior                            |
| **Code & prompt quality**  | 25%    | Comparative | Clean code, thoughtful prompt design, clear README                                             |

> **How comparative scoring works:** Accuracy is scored on an absolute scale: your API is tested against a hidden test suite and judged on correctness. The three subjective dimensions (production feasibility, test quality, code & prompt quality) are evaluated through **direct head-to-head comparison** between submissions. This means your rating in these dimensions reflects how your approach compares to other applicants, not an arbitrary absolute threshold. We use this approach because LLMs are more reliable at saying "A is better than B" than at assigning consistent absolute scores. (Think of how judges at competitions compare entries side by side.)

## What We're Looking For

- **Prompt engineering skill.** Can you get an LLM to reliably produce structured, accurate output?
- **Software craft.** Clean code, proper error handling, good test coverage.
- **Product thinking.** Do your design decisions reflect someone who thinks about the end user (a language learner)?
- **Cost-effectiveness.** Could this run in production without burning money? Model choice, token usage, and caching strategy matter.
- **Communication.** Can you explain your approach clearly in writing?

## When Is My Submission "Done"?

Your submission is ready when:

1. **It works**: `docker compose up` starts cleanly, the health check passes, and `/feedback` returns valid JSON for a variety of inputs (different languages, correct sentences, sentences with errors, non-Latin scripts).
2. **It's tested**: You have a meaningful test suite that you'd be comfortable showing to a colleague.
3. **It's explained**: Your README describes what you built, why you made the choices you did, and how to run it.
4. **You'd ship it**: If someone said "we're deploying this tomorrow," you wouldn't be embarrassed.

Don't chase perfection. A clean, working submission that does the basics well will outperform an ambitious half-finished one. Depth in one area (a great prompt, a thorough test suite, a clever optimization) can set you apart, but only if the foundations are solid first.

Your submission must use **Python** and either **OpenAI** or **Anthropic** as your LLM provider. You are free to use any Python web framework (FastAPI, Flask, Django, etc.) and any OpenAI or Anthropic model. The sample submission uses Python + FastAPI + OpenAI.

> **Important**: The automated scorer runs your tests _inside_ your Docker container via `docker compose exec feedback-api`. Make sure your test dependencies are installed in your Docker image and that your Docker Compose service is named `feedback-api`.

## Example I/O

See [examples/sample_inputs.json](examples/sample_inputs.json) for 5 example input/output pairs covering Spanish, French, Japanese, German, and Portuguese.

## FAQ

**What LLM can I use?**
OpenAI or Anthropic. You can use any model from either provider, and you can chain or combine models from both. Your choice of model is part of your design, so own it and explain it in your README.

**Do I have to use Python / FastAPI / OpenAI?**
Python is required. FastAPI is not: you can use Flask, Django, or any other Python web framework. For LLM providers, you must use OpenAI, Anthropic, or both.

**Do I need to pay for an API key?**
OpenAI and Anthropic both offer free trial credits for new accounts, which should be more than enough for this task.

**Can I use AI tools like ChatGPT or Copilot?**
Yes, fully allowed. See the [AI Use section in RULES.md](RULES.md#ai-use) for our take on this.

**What if my API returns slightly different output than the examples?**
That's fine. The examples show the shape we expect. Exact wording of explanations, corrections, etc. will naturally vary. Your response must conform to the JSON schema, but the linguistic content is judged on accuracy, not exact string matching.

**What if the input sentence is already correct?**
Return `is_correct: true`, an empty `errors` array, and set `corrected_sentence` to the original sentence. This is tested.

**How many languages do I need to support?**
Your API should handle any language a user might submit. The hidden test suite covers 8+ languages including non-Latin scripts (Japanese, Korean, Russian, Chinese). You don't need language-specific logic (the LLM handles this), but your prompt should be robust enough to work across scripts and writing systems.

**Do I need to speak the languages being tested?**
No. The whole point is that the LLM does the linguistic heavy lifting. But you should think about how to verify accuracy for languages you don't know (hint: that's a good thing to discuss in your README).

**Can I modify the JSON schemas?**
No. Your response must conform to the provided `schema/response.schema.json`. You can add extra fields, but the required fields and their types must match.

**What does "replace this README" mean?**
When you submit, your fork's README should describe _your_ approach: your design decisions, prompt strategy, how to run it, and anything interesting you tried. Delete the task description and write your own.

**What if Docker isn't working on my machine?**
Your server must be runnable via `docker compose up`. If you're having Docker issues locally, make sure your Dockerfile works. We will run it in a clean environment. If you need help, that's a fine thing to Google or ask an AI about.

**Is housing provided for the internship?**
No. The internship is on-site in Richmond, Virginia. You are responsible for your own housing and transportation. Richmond has a relatively low cost of living compared to major tech hubs. Summer sublets near VCU and the Fan District are typically $600–900/month.

**I'm an international student with a valid work permit (e.g., CPT/OPT). Am I eligible?**
Yes, as long as you are authorized to work in the United States during the internship period (June–August 2026). We do not sponsor employment visas, but existing work authorization (CPT, OPT, etc.) is fine.

**Will I get feedback on my submission if I'm not selected?**
We can't provide individual feedback to all applicants given the volume, but we may publish general observations about what strong submissions had in common.

**Can I start over after forking?**
Yes. You can `git push --force` as many times as you like before the deadline. We only evaluate what's in your fork at the time the deadline passes.

**Who reviews the submissions?**
Initial scoring is fully automated (Docker build, schema validation, accuracy testing). Accuracy is scored by an LLM judge that runs **twice per submission** and averages the scores to reduce variance. Subjective dimensions (production feasibility, test quality, code quality) are scored through head-to-head comparison between submissions, so your rating reflects how your approach stacks up against the pool. The top submissions are reviewed in full by humans on the Pangea Chat team.

**I'm uncomfortable with automated scoring.**
We understand. We chose transparency over black-box screening: you can see exactly what's evaluated and how. The automated pipeline produces a shortlist; it does not make the hiring decision. Every top submission is reviewed in full by humans on our team.

## Questions?

If something is ambiguous, make a reasonable assumption, state it in your README, and move forward. This is part of the evaluation. We want to see how you handle ambiguity.

---

See [RULES.md](RULES.md) for full assessment rules, eligibility, and legal terms.
