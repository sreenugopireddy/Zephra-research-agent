# Multi-Source Web Research Agent — Phase 1

A search-engine agent that takes a natural-language research question, searches **two
independent web-search providers**, normalizes/deduplicates/ranks what comes back, fetches
the best sources, and produces an **evidence-grounded, citation-validated answer** with a
Groq-hosted LLM.

Phase 1 is a clean, working, testable core backend. No frontend, no auth, no database, no
vector store, no streaming, no autonomous browsing.

---

## Why it's a graph, not an agent loop

The workflow is an explicit LangGraph state machine. There is exactly **one cycle**
(`refine_search → search_providers`) and it is capped by deterministic routing code, not by
the model.

```
START
 → validate_request
 → analyze_query
 → plan_search
 → search_providers ←──────────────┐
 → normalize_results               │
 → deduplicate_results             │
 → rank_sources                    │
 → fetch_sources                   │
 → assess_evidence                 │
 ├── refine_search ────────────────┘   (bounded)
 └── synthesize_answer
 → validate_citations
 → build_response
 → END
```

**Hard bounds** (enforced in `app/graph/routes.py` and `app/config.py`; request options can
lower them but never raise them):

| Bound | Value |
|---|---|
| Search iterations | 2 |
| Queries per iteration | 6 |
| Sources fetched | 8 |
| Sources sent to synthesis | 6 |

`recursion_limit=40` on the compiled graph is a second, independent guard.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env      # then fill in your keys — never commit this file
uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/research \
  -H 'content-type: application/json' \
  -d '{
    "question": "What are the trade-offs between RAG and fine-tuning for enterprise customer support?",
    "options": {"max_iterations": 2, "max_sources": 6, "include_debug": false}
  }'
```

Docker:

```bash
docker compose up --build       # reads .env, serves on :8000
```

Interactive docs at `/docs`, liveness at `/health`.

---

## API

`POST /research`

```json
{
  "question": "...",
  "options": {"max_iterations": 2, "max_sources": 6, "include_debug": false}
}
```

Returns `request_id`, `answer`, `claims[]` (text + source_ids + confidence), `sources[]`,
`conflicts[]`, `uncertainties[]`, `provider_status{}` and `metadata{}`. Set
`include_debug: true` to also get executed queries, provider errors, per-source ranking
breakdowns and the raw citation-validation record.

---

## Design notes

### Providers
`SearchProvider` (`app/providers/base.py`) exposes `search(query) -> list[SearchResult]`.
Tavily and Google run **concurrently**, and each query fans out concurrently within a
provider.

Errors are split so retries are never wasted:

| Family | Cases | Behaviour |
|---|---|---|
| `TransientProviderError` | timeout, connection reset, 408/425/429/500/502/503/504 | retried, exponential backoff, ≤3 attempts |
| `PermanentProviderError` | 401/403 (bad credentials), 400/404, unconfigured provider | never retried |

Each provider has its own timeout. One provider failing never stops the other — its status
and error are recorded in graph state and surfaced in the response. If **both** fail, the
agent returns a clear evidence-unavailable response rather than crashing or inventing one.

> **Note on `langchain-tavily`:** the providers call the documented Tavily and Google REST
> endpoints directly through `httpx`. The wrapper libraries don't cleanly expose the
> per-provider timeout, async call path and precise error classification the retry policy
> needs. Output is plain `SearchResult` models, so providers stay swappable.

### Canonicalization & deduplication
`canonicalize_url` is deterministic: lowercase scheme/host, drop the fragment, drop default
ports, strip a leading `www.`, remove tracking parameters (`utm_*`, `gclid`, `fbclid`,
`msclkid`, …), keep meaningful parameters **sorted**, and normalize the trailing slash
(removed except at the site root).

Sources are then merged on, in priority order: canonical URL → canonical redirect target →
normalized title + domain (titles under 15 characters are too weak to merge on). A merge
preserves `source_id`, `found_by`, `matched_queries`, `duplicate_count`, provider ranks and
scores, and every distinct snippet.

**Two providers returning the same URL is one source, not two** — it raises the
provider-agreement component of the ranking score and nothing else.

### Ranking (deterministic, documented, not LLM-controlled)

```
score = 0.35*relevance
      + 0.20*authority
      + 0.15*freshness
      + 0.10*content_availability
      + 0.20*provider_agreement
```

Each component is in `[0,1]`, so the score is too. Every component is stored in
`score_breakdown` on the ranked source, so any ordering can be audited.

- **relevance** — token overlap between question + matched queries and title + snippet,
  blended with positional decay over the provider's own rank and its score.
- **authority** — tier by domain: `.gov`/`.edu`/standards bodies/arXiv (1.0) > official docs
  (0.9) > reputable technical publications and `.org` (0.7) > default (0.45) >
  content farms and social platforms (0.2).
- **freshness** — recency of `published_date`, and **only weighted when the query actually
  needs current information**; otherwise a neutral 0.5 so evergreen sources aren't punished.
- **content_availability** — snippet length as a proxy for extractable text.
- **provider_agreement** — 1.0 when independent providers both surfaced the source.

The LLM consumes this ordering; it never replaces it.

### Fetching
Every URL passes a safety check first (`app/validation/safety.py`): http(s) only — `file://`,
`data:`, `javascript:` are rejected — plus loopback, link-local, private and reserved
addresses. Pages are fetched concurrently and **independently**: one failure never aborts the
batch. Text extraction uses the standard library only (no extra dependency).

| Status | Meaning |
|---|---|
| `success` | full page text extracted (≥400 chars) |
| `partial` | page unavailable or too thin — only the search snippet is retained |
| `failed` | nothing usable; never cited as fully inspected |

Content is truncated (`MAX_CONTENT_CHARS_PER_SOURCE`) before it can reach Groq.

### Synthesis & citation validation
Synthesis is prompted to use only supplied evidence, cite material claims as `[S1]`, never
invent IDs/URLs/dates/statistics, flag conflicts and gaps, and treat provider agreement as
*not* proof of correctness.

The agent does not trust the model to police itself. `app/validation/citations.py`
deterministically checks that every `[S#]` marker and every `claim.source_ids` entry refers
to a supplied source, that cited sources were actually selected and carry a valid URL, and
that no unknown IDs appear anywhere. On failure:

1. **Retry once** with a stricter prompt naming the invalid IDs and the only permitted IDs.
2. Still failing → degrade safely: bad markers become `[unsupported]`, affected claims are
   kept but marked `unsupported` with bad IDs removed, and `stop_reason` becomes
   `completed_with_invalid_citations_removed`.

Claims resting only on `partial` sources are downgraded from `high` confidence rather than
presented as fully verified.

### Failure behaviour, summarized

| Failure | Result |
|---|---|
| One provider down | Continue on the other; status + error reported |
| Both providers down | Evidence-unavailable response, no invented answer |
| Query analysis fails | Safe fallback: `intent=research`, sub-questions = original question |
| Evidence assessment fails | Proceed to synthesis rather than loop |
| Synthesis fails | `stop_reason=synthesis_unavailable`, safe message, HTTP 200 |
| A page won't fetch | Snippet kept as `partial`; other pages unaffected |

---

## Logging

Structured single-line JSON (`app/observability/logging.py`) covering `request_id`, generated
queries, provider status, result counts, duplicate counts, selected source IDs, fetch
failures, search iteration, citation-validation result and the final stop reason.

Any field whose name suggests a secret (`api_key`, `token`, `secret`, `authorization`, …) is
replaced with `***redacted***`. **API keys are never logged.**

---

## Testing

```bash
pytest -q          # 125 tests, no network, no API keys required
ruff check .
```

External APIs are mocked throughout (`httpx.MockTransport` for providers, injected fakes for
Groq via `set_llm_client`, and `set_providers` for the provider registry).

Covered: URL canonicalization · tracking-parameter removal · duplicate URL merging (including
cross-provider and redirect-target cases) · provider normalization · provider timeout
handling · retry vs no-retry classification · one-provider-down failover · source ranking ·
maximum search-iteration enforcement · invalid citation detection · safe partial response
generation · URL safety.

The integration suite drives the whole graph with mocked Tavily results, Google results,
source fetching and Groq structured responses, and asserts it reaches a **final cited
answer** whose every citation maps to a returned source.

---

## Project structure

```
app/
├── main.py                  FastAPI app: POST /research, GET /health
├── config.py                env-driven settings and hard bounds
├── schemas/                 requests, search, evidence, responses
├── graph/                   state, builder, nodes, routes, runner
├── llm/                     groq_client, prompts, structured_models
├── providers/               base, tavily_provider, google_provider, registry
├── retrieval/               normalize, canonicalize, deduplicate, fetch, rank
├── validation/              citations, safety
└── observability/           logging
tests/
├── unit/  integration/  fixtures/
```

---

## Phase 1 limitations

- **HTML-only extraction.** PDFs and JS-rendered pages fall back to snippet-level `partial`
  evidence. No browser automation in this phase.
- **Heuristic authority list.** The high-authority domain set is hand-maintained; a real
  deployment wants a maintained registry or a learned signal.
- **No caching.** Identical questions re-run every search and fetch; no Redis in Phase 1.
- **No rate limiting or concurrency ceiling** across simultaneous requests.
- **`robots.txt` is not consulted** during fetching.
- **Freshness parsing** relies on provider-supplied dates, which are often missing or wrong;
  publication dates are not independently verified.
- **Citation validation is structural, not semantic.** It proves a cited source exists and
  was retrieved — not that the source actually supports the claim. Semantic entailment
  checking is future work.
- **Conflicts are LLM-identified**, so subtle disagreements can be missed.
- Not yet run against live Tavily/Google/Groq credentials in this environment; the suite
  pins behaviour with mocks.
