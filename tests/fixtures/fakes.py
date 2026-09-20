"""Fakes used across the test suite: providers, LLM and fetched documents."""
from __future__ import annotations

from datetime import UTC, datetime

from app.llm.structured_models import FinalAnswer, SynthesizedClaim
from app.providers.base import SearchProvider
from app.schemas.evidence import EvidenceAssessment, QueryAnalysis, QueryIntent
from app.schemas.search import ExtractionStatus, FetchedDocument, SearchResult


class FakeProvider(SearchProvider):
    """Returns canned results, or raises a canned exception."""

    def __init__(self, name: str, results=None, error: Exception | None = None, enabled=True):
        super().__init__(timeout=1.0, max_results=5)
        self.name = name
        self._results = results or []
        self._error = error
        self._enabled = enabled
        self.calls: list[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _search_once(self, query: str) -> list[SearchResult]:
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        return [r.model_copy(update={"matched_query": query}) for r in self._results]


def tavily_results() -> list[SearchResult]:
    return [
        SearchResult(
            title="RAG versus fine-tuning for enterprise support",
            url="https://arxiv.org/abs/2401.00001?utm_source=newsletter",
            snippet="RAG keeps knowledge external and updatable; fine-tuning bakes behaviour in.",
            published_date="2024-06-01",
            provider="tavily",
            provider_rank=0,
            provider_score=0.93,
        ),
        SearchResult(
            title="Retrieval augmented generation cost analysis",
            url="https://docs.anthropic.com/guides/rag",
            snippet="Operational cost comparison between retrieval and tuning approaches.",
            provider="tavily",
            provider_rank=1,
            provider_score=0.81,
        ),
    ]


def google_results() -> list[SearchResult]:
    return [
        SearchResult(
            # Same source as Tavily's first hit, different URL form -> must merge
            title="RAG versus fine-tuning for enterprise support",
            url="https://www.arxiv.org/abs/2401.00001/",
            snippet="Independent confirmation of the same paper.",
            provider="google",
            provider_rank=0,
        ),
        SearchResult(
            title="Customer support automation benchmarks",
            url="https://example.gov/reports/support-automation",
            snippet="Government benchmark study on support automation quality.",
            published_date="2025-01-15",
            provider="google",
            provider_rank=1,
        ),
    ]


def make_documents(source_ids=("S1", "S2")) -> list[FetchedDocument]:
    return [
        FetchedDocument(
            source_id=sid,
            title=f"Document {sid}",
            url=f"https://example.org/{sid.lower()}",
            domain="example.org",
            content=f"Evidence body for {sid}. " * 30,
            retrieved_at=datetime.now(UTC).isoformat(),
            extraction_status=ExtractionStatus.success,
            content_chars=600,
        )
        for sid in source_ids
    ]


class FakeLLM:
    """Deterministic stand-in for GroqLLMClient."""

    def __init__(
        self,
        analysis: QueryAnalysis | None = None,
        assessments: list[EvidenceAssessment] | None = None,
        answers: list[FinalAnswer] | None = None,
        fail_analysis: bool = False,
    ):
        self.analysis = analysis or QueryAnalysis(
            intent=QueryIntent.comparison,
            requires_current_information=False,
            requires_multiple_queries=True,
            sub_questions=["What is RAG?", "What is fine-tuning?"],
            preferred_source_types=["research_paper"],
            expected_answer_format="prose",
        )
        self.assessments = assessments or [
            EvidenceAssessment(sufficient=True, needs_more_search=False)
        ]
        self.answers = answers or [
            FinalAnswer(
                answer_markdown="RAG keeps knowledge external [S1]; fine-tuning embeds it [S2].",
                claims=[
                    SynthesizedClaim(
                        text="RAG keeps knowledge external.", source_ids=["S1"], confidence="high"
                    )
                ],
                conflicts=[],
                uncertainties=[],
                source_ids_used=["S1", "S2"],
            )
        ]
        self.fail_analysis = fail_analysis
        self.assess_calls = 0
        self.synth_calls = 0

    async def analyze_query(self, question: str) -> QueryAnalysis:
        if self.fail_analysis:
            raise RuntimeError("groq unavailable")
        return self.analysis

    async def assess_evidence(self, **kwargs) -> EvidenceAssessment:
        index = min(self.assess_calls, len(self.assessments) - 1)
        self.assess_calls += 1
        return self.assessments[index]

    async def synthesize(self, **kwargs) -> FinalAnswer:
        index = min(self.synth_calls, len(self.answers) - 1)
        self.synth_calls += 1
        return self.answers[index]
