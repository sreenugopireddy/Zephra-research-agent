"""LangGraph nodes. Each node is small, typed, and owns one step."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from app.config import get_settings
from app.graph.state import ResearchState
from app.llm.groq_client import get_llm_client
from app.llm.structured_models import FinalAnswer
from app.observability.logging import log_event
from app.providers.base import ProviderError, SearchProvider
from app.providers.registry import get_providers
from app.retrieval.deduplicate import deduplicate_results
from app.retrieval.fetch import fetch_documents
from app.retrieval.normalize import normalize_results
from app.retrieval.rank import rank_sources
from app.schemas.evidence import EvidenceAssessment, QueryAnalysis, QueryIntent, SearchPlan
from app.schemas.responses import (
    Claim,
    ResearchMetadata,
    ResearchResponse,
    SourceRef,
)
from app.schemas.search import ProviderStatus, SearchResult
from app.validation.citations import (
    CitationValidation,
    sanitize_answer,
    to_response_claims,
    validate_citations,
)

MAX_QUESTION_CHARS = 2000


# --------------------------------------------------------------------------
# 1. validate_request
# --------------------------------------------------------------------------
async def validate_request(state: ResearchState) -> dict[str, Any]:
    question = (state.get("user_question") or "").strip()
    if len(question) < 3:
        return {"fatal_error": "question is empty or too short", "stop_reason": "invalid_request"}
    if len(question) > MAX_QUESTION_CHARS:
        return {
            "fatal_error": "question exceeds the maximum length",
            "stop_reason": "invalid_request",
        }
    log_event("request_validated", request_id=state["request_id"], question_chars=len(question))
    return {"user_question": question, "fatal_error": None}


# --------------------------------------------------------------------------
# 2. analyze_query
# --------------------------------------------------------------------------
def _fallback_analysis(question: str) -> QueryAnalysis:
    return QueryAnalysis(
        intent=QueryIntent.research,
        requires_current_information=False,
        requires_multiple_queries=False,
        sub_questions=[question],
        preferred_source_types=[],
        expected_answer_format="prose",
    )


async def analyze_query(state: ResearchState) -> dict[str, Any]:
    question = state["user_question"]
    try:
        analysis = await get_llm_client().analyze_query(question)
        if not analysis.sub_questions:
            analysis.sub_questions = [question]
    except Exception as exc:
        log_event(
            "query_analysis_fallback",
            level="warning",
            request_id=state["request_id"],
            error=type(exc).__name__,
        )
        analysis = _fallback_analysis(question)

    intent = getattr(analysis.intent, "value", analysis.intent)
    log_event("query_analyzed", request_id=state["request_id"], intent=intent)
    return {
        "query_intent": intent,
        "freshness_required": analysis.requires_current_information,
        "sub_questions": analysis.sub_questions[:5],
        "preferred_source_types": analysis.preferred_source_types,
        "expected_answer_format": analysis.expected_answer_format,
    }


# --------------------------------------------------------------------------
# 3. plan_search
# --------------------------------------------------------------------------
_SOURCE_TYPE_HINTS = {
    "official_documentation": "official documentation",
    "research_paper": "research paper",
    "government": "site:gov",
    "standards": "specification",
}


def _dedupe_preserving_order(values: Sequence[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = " ".join((value or "").split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def build_search_plan(
    question: str,
    sub_questions: Sequence[str],
    intent: str,
    preferred_source_types: Sequence[str],
    max_queries: int,
    max_sources_to_fetch: int,
    already_executed: Sequence[str] = (),
) -> SearchPlan:
    """Deterministically derive bounded queries. The original question is kept."""
    candidates: list[str] = [question]
    candidates.extend(sub_questions)

    if intent in {"comparison", "multi_part"}:
        candidates.append(f"{question} trade-offs")
        candidates.append(f"{question} comparison evidence")
    if intent == "how_to":
        candidates.append(f"{question} step by step guide")
    if intent == "current_event":
        candidates.append(f"{question} latest")

    for source_type in preferred_source_types:
        hint = _SOURCE_TYPE_HINTS.get(source_type)
        if hint:
            candidates.append(f"{question} {hint}")

    executed = {q.lower() for q in already_executed}
    fresh = [q for q in candidates if q.lower() not in executed]
    queries = _dedupe_preserving_order(fresh or candidates[:1], max_queries)
    return SearchPlan(
        queries=queries,
        provider_strategy="parallel_all_providers",
        required_source_count=3,
        maximum_sources_to_fetch=max_sources_to_fetch,
    )


async def plan_search(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    options = state.get("options") or {}
    plan = build_search_plan(
        question=state["user_question"],
        sub_questions=state.get("sub_questions") or [],
        intent=state.get("query_intent") or "research",
        preferred_source_types=state.get("preferred_source_types") or [],
        max_queries=settings.max_queries_per_iteration,
        max_sources_to_fetch=options.get("max_sources_to_fetch", settings.max_sources_to_fetch),
    )
    log_event(
        "search_planned",
        request_id=state["request_id"],
        iteration=state.get("search_iteration", 0) + 1,
        queries=plan.queries,
    )
    return {
        "search_queries": plan.queries,
        "search_iteration": state.get("search_iteration", 0) + 1,
    }


# --------------------------------------------------------------------------
# 4. search_providers
# --------------------------------------------------------------------------
async def _run_provider(
    provider: SearchProvider, queries: Sequence[str]
) -> tuple[str, list[SearchResult], str, str]:
    """Run one provider across all queries. Returns (name, results, status, error)."""
    if not provider.enabled:
        return provider.name, [], ProviderStatus.skipped.value, "provider not configured"

    outcomes = await asyncio.gather(
        *(provider.search(query) for query in queries), return_exceptions=True
    )
    results: list[SearchResult] = []
    errors: list[str] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            if isinstance(outcome, ProviderError):
                errors.append(str(outcome))
            else:
                errors.append(f"{type(outcome).__name__}: {outcome}")
            continue
        results.extend(outcome)

    if results and errors:
        return provider.name, results, ProviderStatus.partial.value, "; ".join(errors[:3])
    if results:
        return provider.name, results, ProviderStatus.success.value, ""
    if errors:
        return provider.name, [], ProviderStatus.failed.value, "; ".join(errors[:3])
    return provider.name, [], ProviderStatus.success.value, ""


async def search_providers(state: ResearchState) -> dict[str, Any]:
    queries = state.get("search_queries") or []
    providers = get_providers()

    outcomes = await asyncio.gather(
        *(_run_provider(p, queries) for p in providers), return_exceptions=True
    )

    status = dict(state.get("provider_status") or {})
    errors = dict(state.get("provider_errors") or {})
    raw: list[SearchResult] = list(state.get("raw_results") or [])

    for provider, outcome in zip(providers, outcomes, strict=False):
        if isinstance(outcome, BaseException):
            status[provider.name] = ProviderStatus.failed.value
            errors[provider.name] = f"{type(outcome).__name__}: {outcome}"
            continue
        name, results, provider_status, error = outcome
        status[name] = provider_status
        if error:
            errors[name] = error
        raw.extend(results)

    executed = list(state.get("executed_queries") or []) + list(queries)
    log_event(
        "providers_searched",
        request_id=state["request_id"],
        iteration=state.get("search_iteration"),
        provider_status=status,
        result_count=len(raw),
    )
    return {
        "raw_results": raw,
        "provider_status": status,
        "provider_errors": errors,
        "executed_queries": executed,
    }


# --------------------------------------------------------------------------
# 5-7. normalize / deduplicate / rank
# --------------------------------------------------------------------------
async def normalize_results_node(state: ResearchState) -> dict[str, Any]:
    normalized = normalize_results(state.get("raw_results") or [])
    log_event(
        "results_normalized",
        request_id=state["request_id"],
        raw=len(state.get("raw_results") or []),
        normalized=len(normalized),
    )
    return {"normalized_results": normalized}


async def deduplicate_results_node(state: ResearchState) -> dict[str, Any]:
    normalized = state.get("normalized_results") or []
    sources = deduplicate_results(normalized)
    merged = len(normalized) - len(sources)
    log_event(
        "results_deduplicated",
        request_id=state["request_id"],
        unique_sources=len(sources),
        duplicates_merged=merged,
    )
    return {"deduplicated_sources": sources}


async def rank_sources_node(state: ResearchState) -> dict[str, Any]:
    ranked = rank_sources(
        state.get("deduplicated_sources") or [],
        question=state["user_question"],
        freshness_required=bool(state.get("freshness_required")),
        preferred_source_types=state.get("preferred_source_types") or [],
    )
    log_event(
        "sources_ranked",
        request_id=state["request_id"],
        top_source_ids=[s.source_id for s in ranked[:8]],
    )
    return {"ranked_sources": ranked}


# --------------------------------------------------------------------------
# 8. fetch_sources
# --------------------------------------------------------------------------
async def fetch_sources(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    options = state.get("options") or {}
    limit = min(
        int(options.get("max_sources_to_fetch", settings.max_sources_to_fetch)),
        settings.max_sources_to_fetch,
    )
    already = {doc.source_id for doc in state.get("fetched_documents") or []}
    targets = [s for s in (state.get("ranked_sources") or []) if s.source_id not in already][:limit]

    documents = await fetch_documents(
        targets,
        max_chars=settings.max_content_chars_per_source,
        timeout=settings.request_timeout_seconds,
    )
    merged = list(state.get("fetched_documents") or []) + documents
    failures = [d.source_id for d in documents if not d.usable]
    log_event(
        "sources_fetched",
        request_id=state["request_id"],
        fetched=len(documents),
        fetch_failures=failures,
    )
    return {"fetched_documents": merged}


# --------------------------------------------------------------------------
# 9. assess_evidence
# --------------------------------------------------------------------------
async def assess_evidence(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    documents = [d for d in (state.get("fetched_documents") or []) if d.usable]
    max_iterations = min(
        int((state.get("options") or {}).get("max_iterations", settings.max_search_iterations)),
        settings.max_search_iterations,
    )

    if not documents:
        assessment = EvidenceAssessment(
            sufficient=False,
            missing_information=["No source content could be retrieved."],
            needs_more_search=state.get("search_iteration", 1) < max_iterations,
            refined_queries=[],
        )
    else:
        try:
            assessment = await get_llm_client().assess_evidence(
                question=state["user_question"],
                sub_questions=state.get("sub_questions") or [],
                documents=documents,
                iteration=state.get("search_iteration", 1),
                max_iterations=max_iterations,
            )
        except Exception as exc:
            log_event(
                "evidence_assessment_fallback",
                level="warning",
                request_id=state["request_id"],
                error=type(exc).__name__,
            )
            assessment = EvidenceAssessment(
                sufficient=True, needs_more_search=False, covered_sub_questions=[]
            )

    log_event(
        "evidence_assessed",
        request_id=state["request_id"],
        iteration=state.get("search_iteration"),
        sufficient=assessment.sufficient,
        needs_more_search=assessment.needs_more_search,
    )
    return {
        "evidence_assessment": assessment,
        "conflicts": list(assessment.conflicts),
        "uncertainties": list(assessment.missing_information),
    }


# --------------------------------------------------------------------------
# 10. refine_search
# --------------------------------------------------------------------------
async def refine_search(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    assessment = state.get("evidence_assessment")
    executed = state.get("executed_queries") or []
    refined = list(getattr(assessment, "refined_queries", []) or [])
    gaps = list(getattr(assessment, "missing_information", []) or [])

    candidates = refined + [f"{state['user_question']} {gap}" for gap in gaps]
    queries = _dedupe_preserving_order(
        [q for q in candidates if q.lower() not in {e.lower() for e in executed}],
        settings.max_queries_per_iteration,
    )
    if not queries:
        queries = _dedupe_preserving_order(
            [f"{state['user_question']} detailed analysis"], settings.max_queries_per_iteration
        )

    log_event(
        "search_refined",
        request_id=state["request_id"],
        iteration=state.get("search_iteration", 1) + 1,
        queries=queries,
    )
    return {"search_queries": queries, "search_iteration": state.get("search_iteration", 1) + 1}


# --------------------------------------------------------------------------
# 11. synthesize_answer
# --------------------------------------------------------------------------
def _selected_documents(state: ResearchState) -> list:
    settings = get_settings()
    options = state.get("options") or {}
    limit = min(
        int(options.get("max_sources", settings.max_sources_for_synthesis)),
        settings.max_sources_for_synthesis,
    )
    order = {s.source_id: i for i, s in enumerate(state.get("ranked_sources") or [])}
    usable = [d for d in (state.get("fetched_documents") or []) if d.usable]
    usable.sort(key=lambda d: order.get(d.source_id, 10**6))
    return usable[:limit]


async def synthesize_answer(state: ResearchState) -> dict[str, Any]:
    documents = _selected_documents(state)
    if not documents:
        return {
            "draft_answer": None,
            "stop_reason": "evidence_unavailable",
            "synthesis_attempts": state.get("synthesis_attempts", 0),
        }

    assessment = state.get("evidence_assessment")
    try:
        answer = await get_llm_client().synthesize(
            question=state["user_question"],
            documents=documents,
            known_conflicts=getattr(assessment, "conflicts", []) or [],
            missing_information=getattr(assessment, "missing_information", []) or [],
        )
    except Exception as exc:
        log_event(
            "synthesis_failed",
            level="error",
            request_id=state["request_id"],
            error=type(exc).__name__,
        )
        return {
            "draft_answer": None,
            "stop_reason": "synthesis_unavailable",
            "synthesis_attempts": state.get("synthesis_attempts", 0) + 1,
        }
    return {
        "draft_answer": answer,
        "synthesis_attempts": state.get("synthesis_attempts", 0) + 1,
    }


# --------------------------------------------------------------------------
# 12. validate_citations
# --------------------------------------------------------------------------
async def validate_citations_node(state: ResearchState) -> dict[str, Any]:
    documents = _selected_documents(state)
    answer: FinalAnswer | None = state.get("draft_answer")
    if answer is None:
        return {"citation_validation": None}

    validation = validate_citations(answer, documents)

    # One stricter retry before degrading the response.
    if not validation.valid and state.get("synthesis_attempts", 1) < 2:
        log_event(
            "citation_validation_retry",
            level="warning",
            request_id=state["request_id"],
            reason=validation.reason,
        )
        try:
            retry = await get_llm_client().synthesize(
                question=state["user_question"],
                documents=documents,
                known_conflicts=state.get("conflicts") or [],
                missing_information=state.get("uncertainties") or [],
                invalid_ids=validation.unknown_ids + validation.invalid_url_ids,
            )
            retry_validation = validate_citations(retry, documents)
            log_event(
                "citation_validated",
                request_id=state["request_id"],
                valid=retry_validation.valid,
                attempt=2,
            )
            return {
                "draft_answer": retry,
                "citation_validation": retry_validation,
                "synthesis_attempts": state.get("synthesis_attempts", 1) + 1,
            }
        except Exception as exc:
            log_event(
                "citation_retry_failed",
                level="error",
                request_id=state["request_id"],
                error=type(exc).__name__,
            )

    log_event(
        "citation_validated",
        request_id=state["request_id"],
        valid=validation.valid,
        unknown_ids=validation.unknown_ids,
        attempt=1,
    )
    return {"citation_validation": validation}


# --------------------------------------------------------------------------
# 13. build_response (terminal node)
# --------------------------------------------------------------------------
_NO_EVIDENCE_TEMPLATE = (
    "No usable evidence could be retrieved for this question, so no answer is given.\n\n"
    "Provider status: {status}.\n\n"
    "This is an evidence-unavailable result, not a statement about the question itself."
)


def _source_refs(state: ResearchState, documents) -> list[SourceRef]:
    by_id = {s.source_id: s for s in (state.get("ranked_sources") or [])}
    refs: list[SourceRef] = []
    for doc in documents:
        source = by_id.get(doc.source_id)
        refs.append(
            SourceRef(
                source_id=doc.source_id,
                title=doc.title or (source.title if source else ""),
                url=doc.url,
                provider=", ".join(source.found_by) if source else "",
                domain=doc.domain or (source.domain if source else ""),
                extraction_status=getattr(doc.extraction_status, "value", doc.extraction_status),
            )
        )
    return refs


async def build_response(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    documents = _selected_documents(state)
    answer: FinalAnswer | None = state.get("draft_answer")
    validation: CitationValidation | None = state.get("citation_validation")

    conflicts = list(state.get("conflicts") or [])
    uncertainties = list(state.get("uncertainties") or [])
    stop_reason = state.get("stop_reason") or "completed"
    claims: list[Claim] = []

    if state.get("fatal_error"):
        answer_text = f"Request rejected: {state['fatal_error']}"
        stop_reason = state.get("stop_reason") or "invalid_request"
    elif answer is None:
        answer_text = _NO_EVIDENCE_TEMPLATE.format(
            status=", ".join(
                f"{k}={v}" for k, v in (state.get("provider_status") or {}).items()
            )
            or "no providers ran"
        )
        uncertainties.append("No source content was available to answer this question.")
        stop_reason = stop_reason if stop_reason != "completed" else "evidence_unavailable"
    elif validation is not None and not validation.valid:
        answer_text, claims = sanitize_answer(answer, validation)
        conflicts.extend(answer.conflicts)
        uncertainties.extend(answer.uncertainties)
        uncertainties.append(f"Citation validation failed: {validation.reason}")
        stop_reason = "completed_with_invalid_citations_removed"
    else:
        answer_text = answer.answer_markdown
        claims = to_response_claims(answer.claims)
        conflicts.extend(answer.conflicts)
        uncertainties.extend(answer.uncertainties)

    metadata = ResearchMetadata(
        search_iterations=state.get("search_iteration", 0),
        sources_considered=len(state.get("deduplicated_sources") or []),
        sources_selected=len(documents),
        queries_executed=len(state.get("executed_queries") or []),
        duplicates_merged=max(
            0,
            len(state.get("normalized_results") or [])
            - len(state.get("deduplicated_sources") or []),
        ),
        citation_validation=(
            "not_run" if validation is None else ("passed" if validation.valid else "failed")
        ),
        stop_reason=stop_reason,
        elapsed_seconds=round(time.monotonic() - state.get("started_at", time.monotonic()), 3),
    )

    response = ResearchResponse(
        request_id=state["request_id"],
        answer=answer_text,
        claims=claims,
        sources=_source_refs(state, documents),
        conflicts=_dedupe_preserving_order(conflicts, 20),
        uncertainties=_dedupe_preserving_order(uncertainties, 20),
        provider_status={
            **{p.name: ProviderStatus.not_run.value for p in get_providers()},
            **(state.get("provider_status") or {}),
            "groq": "success" if answer is not None else "unavailable",
        },
        metadata=metadata,
    )

    if (state.get("options") or {}).get("include_debug"):
        response.debug = {
            "executed_queries": state.get("executed_queries") or [],
            "provider_errors": state.get("provider_errors") or {},
            "ranked": [
                {"source_id": s.source_id, "score": s.score, "breakdown": s.score_breakdown}
                for s in (state.get("ranked_sources") or [])[: settings.max_sources_to_fetch]
            ],
            "citation_validation": validation.model_dump() if validation else None,
        }

    log_event(
        "request_completed",
        request_id=state["request_id"],
        stop_reason=stop_reason,
        selected_source_ids=[s.source_id for s in response.sources],
        citation_validation=metadata.citation_validation,
        search_iteration=metadata.search_iterations,
    )
    return {"final_response": response, "stop_reason": stop_reason}
