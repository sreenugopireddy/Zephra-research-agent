"""End-to-end graph tests. Tavily, Google, page fetching and Groq are all mocked."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.graph import nodes
from app.graph.runner import run_research
from app.llm.groq_client import set_llm_client
from app.llm.structured_models import FinalAnswer, SynthesizedClaim
from app.main import app
from app.providers.base import PermanentProviderError, TransientProviderError
from app.providers.registry import set_providers
from app.schemas.evidence import EvidenceAssessment
from app.schemas.requests import ResearchOptions, ResearchRequest
from app.schemas.search import ExtractionStatus, FetchedDocument
from tests.fixtures.fakes import FakeLLM, FakeProvider, google_results, tavily_results

QUESTION = "What are the trade-offs between RAG and fine-tuning for enterprise customer support?"


@pytest.fixture
def mock_fetch(monkeypatch):
    """Mock the fetch stage so no HTTP request is ever made."""

    async def fake_fetch_documents(sources, max_chars=6000, timeout=15.0):
        return [
            FetchedDocument(
                source_id=s.source_id,
                title=s.title,
                url=s.url,
                domain=s.domain,
                content=f"Full retrieved text for {s.source_id}. " * 40,
                retrieved_at="2026-01-01T00:00:00+00:00",
                extraction_status=ExtractionStatus.success,
                content_chars=800,
            )
            for s in sources
        ]

    monkeypatch.setattr(nodes, "fetch_documents", fake_fetch_documents)
    return fake_fetch_documents


def both_providers():
    return [
        FakeProvider("tavily", results=tavily_results()),
        FakeProvider("google", results=google_results()),
    ]


async def test_graph_reaches_a_final_cited_answer(mock_fetch):
    set_providers(both_providers())
    set_llm_client(FakeLLM())

    response = await run_research(ResearchRequest(question=QUESTION))

    # a real, cited answer
    assert response.answer and "[S1]" in response.answer
    assert response.claims and response.claims[0].source_ids == ["S1"]
    assert response.metadata.citation_validation == "passed"
    assert response.metadata.stop_reason == "completed"

    # every cited id maps to a returned source
    returned_ids = {s.source_id for s in response.sources}
    for claim in response.claims:
        assert set(claim.source_ids) <= returned_ids

    # provider status is reported for both providers and groq
    assert response.provider_status["tavily"] == "success"
    assert response.provider_status["google"] == "success"
    assert response.provider_status["groq"] == "success"

    # cross-provider duplicate was merged, not double-counted
    assert response.metadata.duplicates_merged >= 1
    assert response.metadata.sources_considered == 3
    assert response.metadata.search_iterations == 1
    assert len(response.sources) <= 6


async def test_bounds_are_enforced_end_to_end(mock_fetch):
    """Two iterations max, six sources to synthesis max, even if the LLM asks for more."""
    set_providers(both_providers())
    llm = FakeLLM(
        assessments=[
            EvidenceAssessment(
                sufficient=False, needs_more_search=True, refined_queries=["another angle"]
            ),
            EvidenceAssessment(
                sufficient=False, needs_more_search=True, refined_queries=["and another"]
            ),
            EvidenceAssessment(sufficient=False, needs_more_search=True),
        ]
    )
    set_llm_client(llm)

    response = await run_research(ResearchRequest(question=QUESTION))

    assert response.metadata.search_iterations == 2, "iteration cap must stop the loop"
    assert llm.assess_calls == 2
    assert response.metadata.sources_selected <= 6
    assert response.answer


async def test_one_provider_down_still_produces_an_answer(mock_fetch):
    set_providers(
        [
            FakeProvider("tavily", error=TransientProviderError("upstream timeout")),
            FakeProvider("google", results=google_results()),
        ]
    )
    set_llm_client(FakeLLM())

    response = await run_research(ResearchRequest(question=QUESTION))

    assert response.provider_status["tavily"] == "failed"
    assert response.provider_status["google"] == "success"
    assert response.sources, "the surviving provider must still yield sources"
    assert response.answer


async def test_both_providers_down_returns_evidence_unavailable(mock_fetch):
    set_providers(
        [
            FakeProvider("tavily", error=TransientProviderError("timeout")),
            FakeProvider("google", error=PermanentProviderError("invalid credentials")),
        ]
    )
    set_llm_client(FakeLLM())

    response = await run_research(ResearchRequest(question=QUESTION))

    assert response.metadata.stop_reason == "evidence_unavailable"
    assert response.sources == []
    assert response.claims == []
    assert "No usable evidence" in response.answer
    assert response.uncertainties
    assert response.provider_status["groq"] == "unavailable"


async def test_invalid_citations_trigger_a_strict_retry_that_succeeds(mock_fetch):
    set_providers(both_providers())
    bad = FinalAnswer(
        answer_markdown="Claim with a hallucinated citation [S99].",
        claims=[SynthesizedClaim(text="bad", source_ids=["S99"], confidence="high")],
        source_ids_used=["S99"],
    )
    good = FinalAnswer(
        answer_markdown="Corrected claim [S1].",
        claims=[SynthesizedClaim(text="good", source_ids=["S1"], confidence="high")],
        source_ids_used=["S1"],
    )
    llm = FakeLLM(answers=[bad, good])
    set_llm_client(llm)

    response = await run_research(ResearchRequest(question=QUESTION))

    assert llm.synth_calls == 2, "a failed validation must trigger exactly one stricter retry"
    assert response.metadata.citation_validation == "passed"
    assert "[S99]" not in response.answer
    assert response.claims[0].source_ids == ["S1"]


async def test_persistently_invalid_citations_degrade_safely(mock_fetch):
    set_providers(both_providers())
    bad = FinalAnswer(
        answer_markdown="Still hallucinating [S99].",
        claims=[SynthesizedClaim(text="bad", source_ids=["S99"], confidence="high")],
        source_ids_used=["S99"],
    )
    llm = FakeLLM(answers=[bad, bad])
    set_llm_client(llm)

    response = await run_research(ResearchRequest(question=QUESTION))

    assert llm.synth_calls == 2
    assert response.metadata.citation_validation == "failed"
    assert response.metadata.stop_reason == "completed_with_invalid_citations_removed"
    assert "[S99]" not in response.answer
    assert response.claims[0].confidence == "unsupported"
    assert response.claims[0].source_ids == []
    assert any("Citation validation failed" in u for u in response.uncertainties)


async def test_groq_outage_falls_back_without_crashing(mock_fetch):
    set_providers(both_providers())

    class DeadLLM(FakeLLM):
        async def analyze_query(self, question):
            raise RuntimeError("groq down")

        async def assess_evidence(self, **kwargs):
            raise RuntimeError("groq down")

        async def synthesize(self, **kwargs):
            raise RuntimeError("groq down")

    set_llm_client(DeadLLM())
    response = await run_research(ResearchRequest(question=QUESTION))

    assert response.metadata.stop_reason == "synthesis_unavailable"
    assert response.provider_status["groq"] == "unavailable"
    assert response.answer  # a safe message, not a crash


async def test_debug_option_exposes_ranking_and_queries(mock_fetch):
    set_providers(both_providers())
    set_llm_client(FakeLLM())

    response = await run_research(
        ResearchRequest(question=QUESTION, options=ResearchOptions(include_debug=True))
    )
    assert response.debug is not None
    assert response.debug["executed_queries"]
    assert response.debug["ranked"][0]["breakdown"]


async def test_max_sources_option_is_respected(mock_fetch):
    set_providers(both_providers())
    set_llm_client(FakeLLM())
    response = await run_research(
        ResearchRequest(question=QUESTION, options=ResearchOptions(max_sources=1))
    )
    assert len(response.sources) == 1


# --- HTTP layer -----------------------------------------------------------
def test_research_endpoint_returns_the_documented_shape(mock_fetch):
    set_providers(both_providers())
    set_llm_client(FakeLLM())

    with TestClient(app) as client:
        result = client.post("/research", json={"question": QUESTION, "options": {}})

    assert result.status_code == 200
    body = result.json()
    for key in (
        "request_id",
        "answer",
        "claims",
        "sources",
        "conflicts",
        "uncertainties",
        "provider_status",
        "metadata",
    ):
        assert key in body
    assert body["sources"][0]["source_id"] == "S1"
    assert "tavily" in body["provider_status"]
    assert set(body["metadata"]) >= {
        "search_iterations",
        "sources_considered",
        "sources_selected",
    }


def test_endpoint_rejects_an_empty_question():
    with TestClient(app) as client:
        assert client.post("/research", json={"question": ""}).status_code == 422


def test_endpoint_rejects_out_of_range_options():
    with TestClient(app) as client:
        response = client.post(
            "/research", json={"question": "a valid question", "options": {"max_iterations": 99}}
        )
    assert response.status_code == 422


def test_health_endpoint_reports_configuration():
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "tavily" in body["providers_configured"]
