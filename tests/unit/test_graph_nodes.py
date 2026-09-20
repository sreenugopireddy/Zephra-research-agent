"""Node-level and routing behaviour, including the iteration cap."""
from __future__ import annotations

import pytest

from app.config import get_settings
from app.graph import nodes
from app.graph.routes import effective_max_iterations, route_after_evidence, route_after_validation
from app.graph.state import initial_state
from app.llm.groq_client import set_llm_client
from app.providers.base import PermanentProviderError, TransientProviderError
from app.providers.registry import set_providers
from app.schemas.evidence import EvidenceAssessment, QueryIntent
from app.schemas.search import ProviderStatus
from tests.fixtures.fakes import FakeLLM, FakeProvider, google_results, tavily_results


def state(**overrides):
    base = initial_state("req-1", "What are the trade-offs between RAG and fine-tuning?", {})
    base.update(overrides)
    return base


# --- validate_request -----------------------------------------------------
async def test_validate_request_accepts_a_normal_question():
    result = await nodes.validate_request(state())
    assert result["fatal_error"] is None


@pytest.mark.parametrize("question", ["", "  ", "ab"])
async def test_validate_request_rejects_short_questions(question):
    result = await nodes.validate_request(state(user_question=question))
    assert result["fatal_error"]
    assert route_after_validation(result) == "invalid"


async def test_validate_request_rejects_overlong_questions():
    result = await nodes.validate_request(state(user_question="x" * 2001))
    assert result["fatal_error"]


# --- analyze_query --------------------------------------------------------
async def test_analyze_query_uses_the_llm():
    set_llm_client(FakeLLM())
    result = await nodes.analyze_query(state())
    assert result["query_intent"] == QueryIntent.comparison.value
    assert result["sub_questions"] == ["What is RAG?", "What is fine-tuning?"]


async def test_analyze_query_falls_back_safely_when_groq_fails():
    set_llm_client(FakeLLM(fail_analysis=True))
    current = state()
    result = await nodes.analyze_query(current)
    assert result["query_intent"] == "research"
    assert result["sub_questions"] == [current["user_question"]]
    assert result["freshness_required"] is False


# --- plan_search ----------------------------------------------------------
def test_plan_keeps_the_original_question_and_bounds_queries():
    plan = nodes.build_search_plan(
        question="Q",
        sub_questions=[f"sub{i}" for i in range(10)],
        intent="comparison",
        preferred_source_types=["official_documentation"],
        max_queries=6,
        max_sources_to_fetch=8,
    )
    assert plan.queries[0] == "Q"
    assert len(plan.queries) == 6


def test_plan_never_emits_duplicate_queries():
    plan = nodes.build_search_plan("Q", ["Q", "q", " Q "], "research", [], 6, 8)
    assert plan.queries == ["Q"]


def test_plan_expands_comparison_questions():
    plan = nodes.build_search_plan("A vs B", [], "comparison", [], 6, 8)
    assert any("trade-offs" in q for q in plan.queries)


def test_plan_skips_already_executed_queries():
    plan = nodes.build_search_plan("Q", ["sub1"], "research", [], 6, 8, already_executed=["Q"])
    assert "Q" not in plan.queries


async def test_plan_search_increments_the_iteration_counter():
    result = await nodes.plan_search(state(sub_questions=["a"], query_intent="research"))
    assert result["search_iteration"] == 1


# --- search_providers -----------------------------------------------------
async def test_one_provider_failing_does_not_stop_the_other():
    set_providers(
        [
            FakeProvider("tavily", error=PermanentProviderError("bad key", provider="tavily")),
            FakeProvider("google", results=google_results()),
        ]
    )
    result = await nodes.search_providers(state(search_queries=["q1"]))

    assert result["provider_status"]["tavily"] == ProviderStatus.failed.value
    assert result["provider_status"]["google"] == ProviderStatus.success.value
    assert result["provider_errors"]["tavily"]
    assert len(result["raw_results"]) == 2


async def test_both_providers_failing_is_recorded_without_crashing():
    set_providers(
        [
            FakeProvider("tavily", error=TransientProviderError("timeout")),
            FakeProvider("google", error=PermanentProviderError("bad key")),
        ]
    )
    result = await nodes.search_providers(state(search_queries=["q1"]))
    assert result["raw_results"] == []
    assert set(result["provider_status"].values()) == {ProviderStatus.failed.value}


async def test_unconfigured_provider_is_skipped_not_failed():
    set_providers([FakeProvider("tavily", enabled=False), FakeProvider("google", results=[])])
    result = await nodes.search_providers(state(search_queries=["q"]))
    assert result["provider_status"]["tavily"] == ProviderStatus.skipped.value


async def test_provider_errors_are_never_silently_ignored():
    set_providers([FakeProvider("tavily", error=TransientProviderError("boom"))])
    result = await nodes.search_providers(state(search_queries=["q"]))
    assert "boom" in result["provider_errors"]["tavily"]


async def test_every_query_is_sent_to_every_enabled_provider():
    tavily = FakeProvider("tavily", results=tavily_results())
    google = FakeProvider("google", results=google_results())
    set_providers([tavily, google])
    await nodes.search_providers(state(search_queries=["q1", "q2"]))
    assert tavily.calls == ["q1", "q2"] and google.calls == ["q1", "q2"]


# --- iteration bounds -----------------------------------------------------
def test_route_refines_when_evidence_is_insufficient_and_budget_remains():
    current = state(
        search_iteration=1,
        evidence_assessment=EvidenceAssessment(sufficient=False, needs_more_search=True),
    )
    assert route_after_evidence(current) == "refine"


def test_route_stops_at_the_configured_iteration_cap():
    current = state(
        search_iteration=2,
        evidence_assessment=EvidenceAssessment(sufficient=False, needs_more_search=True),
    )
    assert route_after_evidence(current) == "synthesize"


def test_route_synthesizes_when_evidence_is_sufficient():
    current = state(
        search_iteration=1,
        evidence_assessment=EvidenceAssessment(sufficient=True, needs_more_search=False),
    )
    assert route_after_evidence(current) == "synthesize"


def test_request_options_cannot_exceed_the_configured_cap():
    settings = get_settings()
    current = state(options={"max_iterations": 99})
    assert effective_max_iterations(current) == settings.max_search_iterations


def test_route_handles_a_missing_assessment():
    assert route_after_evidence(state(search_iteration=1)) == "synthesize"


# --- refine_search --------------------------------------------------------
async def test_refine_uses_gaps_and_avoids_repeating_queries():
    current = state(
        search_iteration=1,
        executed_queries=["original"],
        evidence_assessment=EvidenceAssessment(
            sufficient=False,
            needs_more_search=True,
            refined_queries=["original", "brand new query"],
            missing_information=["cost data"],
        ),
    )
    result = await nodes.refine_search(current)
    assert "original" not in result["search_queries"]
    assert "brand new query" in result["search_queries"]
    assert result["search_iteration"] == 2


async def test_refine_always_produces_at_least_one_query():
    current = state(
        search_iteration=1,
        executed_queries=[],
        evidence_assessment=EvidenceAssessment(sufficient=False, needs_more_search=True),
    )
    result = await nodes.refine_search(current)
    assert len(result["search_queries"]) >= 1


# --- assess / synthesize with no evidence ---------------------------------
async def test_assessment_without_documents_reports_insufficient():
    set_llm_client(FakeLLM())
    result = await nodes.assess_evidence(state(search_iteration=1, fetched_documents=[]))
    assert result["evidence_assessment"].sufficient is False


async def test_synthesis_is_skipped_when_there_is_no_evidence():
    set_llm_client(FakeLLM())
    result = await nodes.synthesize_answer(state(fetched_documents=[], ranked_sources=[]))
    assert result["draft_answer"] is None
    assert result["stop_reason"] == "evidence_unavailable"
