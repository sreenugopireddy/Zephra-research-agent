"""Typed LangGraph state shared by every node."""
from __future__ import annotations

from typing import Any, TypedDict

from app.schemas.evidence import EvidenceAssessment
from app.schemas.responses import ResearchResponse
from app.schemas.search import (
    DeduplicatedSource,
    FetchedDocument,
    NormalizedResult,
    RankedSource,
    SearchResult,
)
from app.validation.citations import CitationValidation


class ResearchState(TypedDict, total=False):
    # request
    request_id: str
    user_question: str
    options: dict[str, Any]
    started_at: float

    # analysis / planning
    query_intent: str
    freshness_required: bool
    sub_questions: list[str]
    preferred_source_types: list[str]
    expected_answer_format: str
    search_queries: list[str]
    executed_queries: list[str]
    search_iteration: int

    # providers
    provider_status: dict[str, str]
    provider_errors: dict[str, str]

    # retrieval
    raw_results: list[SearchResult]
    normalized_results: list[NormalizedResult]
    deduplicated_sources: list[DeduplicatedSource]
    ranked_sources: list[RankedSource]
    fetched_documents: list[FetchedDocument]

    # reasoning
    evidence_assessment: EvidenceAssessment | None
    conflicts: list[str]
    uncertainties: list[str]
    draft_answer: Any | None
    citation_validation: CitationValidation | None
    synthesis_attempts: int

    # output
    final_response: ResearchResponse | None
    fatal_error: str | None
    stop_reason: str
    execution_metadata: dict[str, Any]


def initial_state(request_id: str, question: str, options: dict[str, Any]) -> ResearchState:
    import time

    return ResearchState(
        request_id=request_id,
        user_question=question,
        options=options,
        started_at=time.monotonic(),
        search_iteration=0,
        synthesis_attempts=0,
        sub_questions=[],
        search_queries=[],
        executed_queries=[],
        provider_status={},
        provider_errors={},
        raw_results=[],
        normalized_results=[],
        deduplicated_sources=[],
        ranked_sources=[],
        fetched_documents=[],
        conflicts=[],
        uncertainties=[],
        execution_metadata={},
        stop_reason="completed",
    )
