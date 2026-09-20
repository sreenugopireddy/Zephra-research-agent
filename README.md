# Multi-Source Web Research Agent

A bounded, evidence-grounded search agent that transforms a natural-language research question into a cited answer by querying two independent web-search providers, consolidating and ranking the returned sources, inspecting selected pages, and validating every citation before responding.

Built for the **Zephra AI Track A — Multi-Source Web Research Agent** assignment.

> This repository focuses on correctness, reliability, and transparent engineering decisions rather than unnecessary product surface area. It includes a lightweight optional dashboard for demonstration, while the core deliverable remains the tested research backend.

---

## 1. At a glance

The agent implements the complete research lifecycle:

```text
Research question
      ↓
Query analysis
      ↓
Search planning
      ↓
Tavily + Serper retrieval
      ↓
Normalization and deduplication
      ↓
Deterministic source ranking
      ↓
Source fetching and extraction
      ↓
Evidence assessment
      ↓
Bounded search refinement
      ↓
Groq synthesis
      ↓
Citation validation
      ↓
Grounded response
```

The central design decision is to use a **bounded LangGraph workflow** instead of an unrestricted autonomous loop. The model assists with planning and synthesis, but the application deterministically controls provider failures, retry behavior, source limits, ranking, iteration limits, and citation integrity.

---

## 2. Why this design

A naïve research agent looks like this:

```text
Question → Search API → LLM answer
```

That approach makes it difficult to verify:

- Whether multiple providers were actually used.
- Whether duplicate pages were counted as independent evidence.
- Why a source was selected.
- What happens when a provider fails.
- Whether the final citations point to retrieved sources.
- Whether the answer contains unsupported claims.

This implementation separates the workflow into explicit stages so that each important decision can be inspected, tested, and explained.

The model is used for tasks that require language reasoning:

- Query analysis.
- Search-plan generation.
- Evidence sufficiency assessment.
- Final answer synthesis.

Deterministic application logic controls:

- Provider execution.
- Timeouts and retries.
- URL canonicalization.
- Deduplication.
- Source ranking.
- Source limits.
- Citation validation.
- Failure recovery.

---

## 3. Requirements coverage

| Assignment requirement | Implementation |
|---|---|
| Decompose questions where useful | `analyze_query` node produces `QueryAnalysis.sub_questions` |
| Query at least two providers | `TavilyProvider` and `SerperProvider`, executed independently and concurrently |
| Normalize and merge results | `retrieval/normalize.py` |
| Deduplicate sources | `retrieval/canonicalize.py` and `retrieval/deduplicate.py` |
| Fetch or inspect content | `retrieval/fetch.py` with per-source failure isolation |
| Rank sources explicitly | `retrieval/rank.py` with an auditable weighted heuristic |
| Synthesize cited answers | `synthesize_answer` node using `[S1]`-style references |
| Report uncertainty | `EvidenceAssessment.missing_information` and response `uncertainties[]` |
| Handle failures and retries | Provider error taxonomy, timeouts, bounded exponential retries |
| Separate workflow stages | LangGraph nodes and single-purpose retrieval/validation modules |
| Demonstrate reproducibility | Environment configuration, Docker, mocked tests, and documented execution |

---

## 4. Architecture

### 4.1 Workflow

```text
START
  ↓
validate_request
  ↓
analyze_query
  ↓
plan_search
  ↓
search_providers
  ↓
normalize_results
  ↓
deduplicate_results
  ↓
rank_sources
  ↓
fetch_sources
  ↓
assess_evidence
  ├── refine_search ──────────────┐
  │                               │
  └── synthesize_answer           │
          ↓                      │
     validate_citations          │
          ↓                      │
      build_response             │
          ↓                      │
          END                    │
                                  │
        ←─────────────────────────┘
```

There is only one refinement cycle:

```text
refine_search → search_providers
```

It is controlled by deterministic routing logic rather than by the model. The compiled graph also uses a recursion limit as a second protection against runaway execution.

### 4.2 Hard bounds

Request-level options may reduce these limits but cannot increase them.

| Limit | Default |
|---|---:|
| Maximum search iterations | 2 |
| Maximum generated queries per iteration | 6 |
| Maximum sources fetched | 8 |
| Maximum sources sent to synthesis | 6 |
| Maximum provider retry attempts | 3 |
| Maximum synthesis retry attempts | 1 |

---

## 5. Technology choices

| Technology | Role | Why it was selected |
|---|---|---|
| FastAPI | HTTP API | Async-friendly, lightweight, and integrates naturally with Pydantic |
| LangGraph | Workflow orchestration | Represents state, nodes, and conditional transitions explicitly |
| LangChain | LLM and tool integration | Provides model abstractions, prompts, and structured-output integration |
| Groq | LLM reasoning and synthesis | Low-latency inference with structured response support |
| Tavily | Search provider | LLM-oriented search and source extraction |
| Serper.dev | Independent search provider | Provides a second search index through a simple REST API |
| `httpx` | HTTP client | Async requests, precise timeout handling, and easy mocking |
| Pydantic | Data contracts | Validates API requests, graph state, provider results, and LLM outputs |
| Tenacity | Retry behavior | Supports bounded asynchronous retries with exponential backoff |
| `html.parser` | Basic extraction | Keeps the core implementation lightweight for HTML pages |
| pytest | Testing | Covers deterministic logic, provider behavior, and complete graph execution |
| Docker | Reproducible runtime | Makes local execution consistent across environments |

The implementation intentionally avoids adding a vector database, browser automation, authentication, or a complex frontend because those features do not materially improve the assignment’s core evaluation criteria.

---

## 6. Setup and execution

### Prerequisites

- Python 3.11+
- Docker, optional
- Groq API key
- Tavily API key
- Serper API key

### Local setup

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements-dev.txt
```

Create local configuration:

```bash
cp .env.example .env
```

Then add credentials to `.env`. Never commit this file.

Start the API:

```bash
uvicorn app.main:app --reload
```

Available endpoints:

```text
GET  /health
POST /research
GET  /dashboard
GET  /docs
```

### Docker

```bash
docker compose up --build
```

### Example request

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the trade-offs between RAG and fine-tuning for enterprise customer support?",
    "options": {
      "max_iterations": 2,
      "max_sources": 6,
      "include_debug": false
    }
  }'
```

---

## 7. Configuration

```env
GROQ_API_KEY=
GROQ_MODEL=

TAVILY_API_KEY=
SERPER_API_KEY=

REQUEST_TIMEOUT_SECONDS=15
MAX_SEARCH_ITERATIONS=2
MAX_QUERIES_PER_ITERATION=6
MAX_SOURCES_TO_FETCH=8
MAX_SOURCES_FOR_SYNTHESIS=6
MAX_CONTENT_CHARS_PER_SOURCE=12000
```

The application validates required configuration at startup or when the relevant provider is first used. Credentials are loaded from environment variables and are never stored in source code or logs.

---

## 8. Request and response model

### Request

```json
{
  "question": "What are the trade-offs between RAG and fine-tuning for enterprise customer support?",
  "options": {
    "max_iterations": 2,
    "max_sources": 6,
    "include_debug": false
  }
}
```

### Response

```json
{
  "request_id": "run-123",
  "answer": "RAG is generally preferable when...",
  "claims": [
    {
      "text": "RAG can incorporate updated documents without retraining the base model.",
      "source_ids": ["S1"],
      "confidence": "high"
    }
  ],
  "sources": [
    {
      "source_id": "S1",
      "title": "Example source",
      "url": "[https://example.com/article](https://example.com/article)",
      "provider": "tavily"
    }
  ],
  "conflicts": [],
  "uncertainties": [],
  "provider_status": {
    "tavily": "success",
    "serper": "success",
    "groq": "success"
  },
  "metadata": {
    "search_iterations": 1,
    "sources_considered": 12,
    "sources_selected": 6,
    "duplicates_merged": 3,
    "citation_validation": "passed",
    "stop_reason": "evidence_sufficient",
    "elapsed_ms": 4820
  }
}
```

With `include_debug: true`, the response additionally includes executed queries, provider errors, source-score breakdowns, and citation-validation details.

---

## 9. Provider architecture

All providers conform to the same internal interface:

```text
search(query) → list[SearchResult]
```

A normalized result contains:

```text
SearchResult
├── title
├── url
├── snippet
├── published_date
├── provider
├── provider_rank
├── provider_score
└── matched_query
```

### Tavily and Serper

The providers run independently and preferably concurrently.

```text
Search plan
   ├── Tavily query 1
   ├── Tavily query 2
   ├── Serper query 1
   └── Serper query 2
```

One provider failing never stops the other provider. The response records each provider’s status and error.

### Error classification

| Error family | Examples | Behavior |
|---|---|---|
| Transient | Timeout, connection reset, 408, 429, 500, 502, 503, 504 | Retry with exponential backoff |
| Permanent | Invalid credentials, 400, 401, 403, 404 | Do not retry |
| Configuration | Missing API key or invalid endpoint | Fail provider clearly and continue where possible |

If both providers fail, the system returns a controlled evidence-unavailable response instead of fabricating an answer.

---

## 10. Result normalization and deduplication

### URL canonicalization

The canonicalization process:

- Lowercases the scheme and hostname.
- Removes URL fragments.
- Removes default ports.
- Strips a leading `www.` where appropriate.
- Removes tracking parameters such as `utm_*`, `gclid`, `fbclid`, and `msclkid`.
- Preserves meaningful query parameters in sorted order.
- Normalizes trailing slashes.

### Duplicate merging

Sources are merged in the following order:

1. Canonical URL.
2. Canonical redirect target.
3. Normalized title plus domain, when the title is sufficiently informative.

A merged source preserves:

- `source_id`.
- Providers that found it.
- Matched queries.
- Duplicate count.
- Provider ranks and scores.
- Distinct snippets.
- Redirect metadata.

Two providers returning the same URL are treated as one source. Provider agreement contributes to ranking but is not considered independent factual confirmation.

---

## 11. Source ranking

Ranking is deterministic and auditable:

```text
score =
    0.35 × relevance
  + 0.20 × authority
  + 0.15 × freshness
  + 0.10 × content_availability
  + 0.20 × provider_agreement
```

Every component is normalized to `[0, 1]` and stored in the source’s `score_breakdown`.

### Ranking signals

- **Relevance:** Token overlap between the question, matched query, title, and snippet, combined with provider rank or score.
- **Authority:** Domain and source-type heuristics.
- **Freshness:** Publication date when the question requires current information; otherwise a neutral score.
- **Content availability:** Whether sufficient content can be extracted.
- **Provider agreement:** Whether independent providers surfaced the same canonical source.

The LLM receives the selected ordering but does not replace the ranking policy.

---

## 12. Source fetching

Selected URLs are fetched concurrently and independently.

Every URL passes safety checks before retrieval:

- Only `http` and `https` schemes are permitted.
- `file://`, `data:`, and `javascript:` URLs are rejected.
- Loopback, link-local, private, and reserved IP addresses are blocked.
- Per-request timeouts and response-size limits are applied.

Extraction statuses:

| Status | Meaning |
|---|---|
| `success` | Sufficient page text was extracted |
| `partial` | The page could not be fully extracted; snippet evidence remains |
| `failed` | No usable content was retrieved |

A failed page never gets presented as a fully inspected source. Content is truncated before being sent to Groq.

---

## 13. Evidence assessment

The `assess_evidence` node evaluates whether the current source set is sufficient.

It returns:

```json
{
  "sufficient": false,
  "covered_sub_questions": [
    "What are the strengths of RAG?"
  ],
  "missing_information": [
    "Evidence comparing maintenance and latency costs"
  ],
  "conflicts": [],
  "needs_more_search": true,
  "refined_queries": [
    "RAG versus fine-tuning maintenance latency production systems"
  ]
}
```

The workflow performs another search only when:

- A major sub-question is unsupported.
- The selected sources are too similar.
- The sources are low quality.
- Important claims conflict.
- Retrieved content is insufficient.

The workflow stops after the configured maximum iteration count and reports remaining evidence gaps.

---

## 14. Grounded synthesis

Groq receives only the selected source documents, each tagged with a stable source ID:

```text
SOURCE [S1]
Title: ...
URL: ...
Content:
...

SOURCE [S2]
Title: ...
URL: ...
Content:
...
```

The synthesis prompt requires the model to:

- Use only supplied evidence.
- Cite material factual claims as `[S1]`, `[S2]`, and so on.
- Never invent source IDs, URLs, dates, figures, or quotations.
- Distinguish facts from interpretation.
- Report conflicts.
- Report missing evidence.
- Avoid treating provider agreement as proof of correctness.

The model returns structured output containing:

```text
answer_markdown
claims
conflicts
uncertainties
source_ids_used
```

---

## 15. Citation validation

Prompt instructions alone are not treated as a reliability mechanism.

The deterministic validator checks:

- Every citation ID exists.
- Every cited source was selected and retrieved.
- Every cited source has a valid URL.
- No unknown citation IDs appear.
- Returned references correspond to supplied sources.

If validation fails:

1. The answer is regenerated once with a stricter prompt.
2. If validation fails again, invalid citation markers are removed or replaced with `[unsupported]`.
3. Affected claims are downgraded and the stop reason records the degradation.

Claims based only on partial sources cannot be presented with full confidence.

This ensures that the system does not silently return polished but structurally unsupported citations.

---

## 16. Failure behavior

| Failure | System behavior |
|---|---|
| One provider is unavailable | Continue with the remaining provider and expose status |
| Both providers fail | Return an evidence-unavailable response |
| Query analysis fails | Use the original question as a safe fallback |
| Evidence assessment fails | Proceed to synthesis without an uncontrolled search loop |
| Synthesis fails | Return a safe synthesis-unavailable response |
| A source cannot be fetched | Retain snippet-level evidence when possible |
| Citation validation fails | Retry once, then safely degrade |
| Maximum iteration reached | Return the best available answer with uncertainty |

The system prefers an explicit limitation over an unsupported confident answer.

---

## 17. Logging and observability

The application emits structured JSON logs containing:

- `request_id`.
- Generated queries.
- Provider status and latency.
- Result counts.
- Duplicate counts.
- Selected source IDs.
- Fetch failures.
- Search iteration.
- Citation-validation result.
- Final stop reason.
- Elapsed time.

Potentially sensitive fields such as API keys, tokens, secrets, and authorization headers are redacted before logging.

---

## 18. Optional dashboard

A small same-origin dashboard is available at:

```text
GET /dashboard
```

It uses vanilla HTML, CSS, and JavaScript without a frontend build system. It allows a reviewer to:

- Submit a question.
- Configure iteration and source limits.
- Toggle debug output.
- View the generated answer.
- Inspect claims and citations.
- Review provider status.
- Inspect execution metadata.

The dashboard is intentionally lightweight. It exists to support manual testing and the demo video; the core assignment deliverable is the research backend.

---

## 19. Testing

Run the test suite:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

The test suite uses mocked provider and LLM responses. It does not require API keys or network access.

### Covered behavior

- URL canonicalization.
- Tracking-parameter removal.
- Cross-provider duplicate merging.
- Redirect-target deduplication.
- Provider result normalization.
- Timeout handling.
- Retry versus no-retry classification.
- One-provider failure and failover.
- Source ranking.
- Maximum search-iteration enforcement.
- Invalid citation detection.
- Safe partial responses.
- URL safety validation.
- Structured-output failure handling.

The integration tests execute the complete graph with mocked Tavily, Serper, source-fetching, and Groq responses. They verify that:

```text
Every returned citation maps to a returned source.
```

Failure paths are also tested, including:

- Both providers unavailable.
- Groq unavailable.
- Citation validation failing twice.
- Individual source-fetch failures.

Live-provider testing was performed separately with configured Tavily, Serper, and Groq credentials. Automated tests remain mocked to keep execution fast, reproducible, and safe.

---

## 20. Project structure

```text
app/
├── main.py                  # FastAPI application and endpoints
├── dashboard.py             # Optional self-contained demo interface
├── config.py                # Environment configuration and hard bounds
├── schemas/
│   ├── requests.py
│   ├── search.py
│   ├── evidence.py
│   └── responses.py
├── graph/
│   ├── state.py
│   ├── builder.py
│   ├── nodes.py
│   ├── routes.py
│   └── runner.py
├── llm/
│   ├── groq_client.py
│   ├── prompts.py
│   └── structured_models.py
├── providers/
│   ├── base.py
│   ├── tavily_provider.py
│   ├── serper_provider.py
│   └── registry.py
├── retrieval/
│   ├── normalize.py
│   ├── canonicalize.py
│   ├── deduplicate.py
│   ├── fetch.py
│   └── rank.py
├── validation/
│   ├── citations.py
│   └── safety.py
└── observability/
    └── logging.py

tests/
├── unit/
├── integration/
└── fixtures/
```

---

## 21. Known limitations

- HTML-only extraction. PDFs and JavaScript-rendered pages fall back to snippet-level evidence.
- Authority scoring uses a hand-maintained heuristic.
- There is no response cache, so repeated questions repeat the retrieval process.
- There is no global rate limit or concurrency ceiling across simultaneous API requests.
- The fetcher does not currently evaluate `robots.txt`.
- Freshness depends on provider-supplied publication dates, which may be absent or inaccurate.
- Citation validation is structural, not semantic. It verifies that a citation exists and was retrieved, but does not prove that the cited content entails the claim.
- Conflict detection is assisted by the LLM, so subtle disagreements may be missed.
- Groq model availability can change. `GROQ_MODEL` should be configured explicitly and checked against the models available to the account.
- The project is delivered as a runnable repository and Docker image, not as a permanently hosted public service.

These limitations are intentionally documented rather than hidden. They define the boundary between this focused assignment implementation and a production research platform.

---

## 22. Future improvements

With additional time, the next improvements would be:

1. Semantic claim-to-source entailment checking.
2. A maintained or learned source-authority model.
3. Query and source-content caching.
4. PDF extraction and JavaScript-rendered page support.
5. Startup model-capability discovery for Groq.
6. Per-client rate limits and concurrency controls.
7. Additional specialized providers for news, academic papers, or government data.
8. Persistent run traces for resumable research sessions.
9. Evaluation against a manually labeled research-question benchmark.
10. Cost and latency dashboards for provider and model usage.

---

## 23. Personal implementation

I personally implemented the research pipeline as a bounded, evidence-aware workflow rather than a single search-to-answer chain.

The main implementation areas were:

- Designing the LangGraph state machine and bounded refinement loop.
- Defining provider-independent search-result schemas.
- Implementing Tavily and Serper provider integrations.
- Adding transient/permanent error classification and bounded retries.
- Implementing URL canonicalization and cross-provider deduplication.
- Designing an auditable source-ranking heuristic.
- Adding concurrent, safety-checked source fetching.
- Integrating Groq structured outputs for query analysis, evidence assessment, and synthesis.
- Implementing deterministic citation validation and safe degradation.
- Adding failure-path tests and mocked integration tests.
- Documenting setup, trade-offs, limitations, and future improvements.

The primary trade-off was to avoid unrestricted agent autonomy. A bounded graph adds some orchestration code, but it makes failure handling, source selection, iteration limits, and citation validation explicit and testable.

---

## 24. Final engineering summary

This project is intentionally not a large production system. It is a focused demonstration of how to build a reliable search agent around uncertain external services.

The core guarantee is:

```text
No final answer is returned without:
- Multi-provider retrieval attempt
- Source normalization
- Duplicate handling
- Evidence inspection
- Citation validation
- Explicit uncertainty reporting
```

The most important design principle is:

> Groq performs language reasoning, LangChain and LangGraph coordinate the workflow, external search providers supply evidence, and deterministic application logic enforces reliability.

That separation makes the system easier to test, debug, extend, and defend during an interview.
