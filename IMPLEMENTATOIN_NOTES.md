# Implementation Notes

## My contribution

I designed and implemented a bounded multi-source web research agent for the
Zephra AI technical evaluation.

## Core implementation

I implemented:

- LangGraph workflow orchestration.
- Query analysis and bounded search planning.
- Tavily and Serper provider integrations.
- Provider-independent result schemas.
- URL canonicalization and cross-provider deduplication.
- Deterministic source ranking.
- Concurrent source fetching with per-source failure isolation.
- Groq-based evidence assessment and answer synthesis.
- Citation validation and safe degradation.
- Provider retry and timeout handling.
- Unit and integration tests.
- Dockerized local execution and documentation.

## Key design decisions

I used a bounded workflow rather than an unrestricted autonomous agent loop.
This makes search iterations, source limits, failure handling, and citation
validation explicit and testable.

I kept ranking, deduplication, retry policy, and citation validation
deterministic. The LLM is responsible for language reasoning and synthesis,
but it is not allowed to silently control reliability-critical behavior.

## Trade-offs

The implementation uses heuristic source ranking instead of a learned ranker
because the heuristic is easier to audit and explain for this assignment.

The fetcher supports HTML pages but does not handle every PDF or
JavaScript-rendered page. When extraction fails, the system preserves snippet
evidence as partial evidence and reports the limitation.

Citation validation is structural: it verifies that cited source IDs exist and
were retrieved. It does not yet perform full semantic claim-to-source
entailment checking.

## Testing

The test suite uses mocked provider and LLM responses, so it runs without API
keys or network access. It covers normal execution, provider failures,
timeouts, retries, duplicate results, source-fetch failures, search
iteration limits, and invalid citations.

## Personal implementation statement

This submission reflects my own implementation and engineering decisions. I
used official documentation to understand the APIs and framework behavior,
and I tested the integrated workflow with mocked and live-provider paths.
