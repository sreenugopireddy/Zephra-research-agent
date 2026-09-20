"""Conditional routing. All loop bounds are enforced here, not by the LLM."""
from __future__ import annotations

from app.config import get_settings
from app.graph.state import ResearchState
from app.observability.logging import log_event


def route_after_validation(state: ResearchState) -> str:
    """Stop immediately on an invalid request."""
    return "invalid" if state.get("fatal_error") else "valid"


def effective_max_iterations(state: ResearchState) -> int:
    """Requested iterations, hard-capped by the configured maximum."""
    settings = get_settings()
    requested = int(
        (state.get("options") or {}).get("max_iterations", settings.max_search_iterations)
    )
    return max(1, min(requested, settings.max_search_iterations))


def route_after_evidence(state: ResearchState) -> str:
    """Decide between one more bounded search round and final synthesis.

    A refinement requires ALL of:
      * the assessment asked for more search,
      * the evidence is not already sufficient,
      * the iteration cap has not been reached.
    """
    assessment = state.get("evidence_assessment")
    iteration = state.get("search_iteration", 1)
    max_iterations = effective_max_iterations(state)

    if assessment is None:
        return "synthesize"
    if iteration >= max_iterations:
        log_event(
            "iteration_cap_reached",
            request_id=state.get("request_id"),
            search_iteration=iteration,
            max_iterations=max_iterations,
        )
        return "synthesize"
    if assessment.needs_more_search and not assessment.sufficient:
        return "refine"
    return "synthesize"
