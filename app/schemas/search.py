"""Schemas for search results as they move through the retrieval pipeline."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ProviderStatus(StrEnum):
    success = "success"
    partial = "partial"
    failed = "failed"
    skipped = "skipped"
    not_run = "not_run"


class ExtractionStatus(StrEnum):
    success = "success"
    partial = "partial"
    failed = "failed"


class SearchResult(BaseModel):
    """A single result exactly as returned by one provider for one query."""

    title: str = ""
    url: str
    snippet: str = ""
    published_date: str | None = None
    provider: str
    provider_rank: int = 0
    provider_score: float | None = None
    matched_query: str = ""
    redirect_target: str | None = None


class NormalizedResult(SearchResult):
    """A provider result mapped into the common schema, with provenance kept."""

    canonical_url: str
    domain: str


class DeduplicatedSource(BaseModel):
    """One real-world source, possibly discovered several times."""

    source_id: str
    title: str = ""
    url: str
    canonical_url: str
    domain: str
    snippets: list[str] = Field(default_factory=list)
    published_date: str | None = None
    found_by: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)
    provider_ranks: dict[str, int] = Field(default_factory=dict)
    provider_scores: dict[str, float] = Field(default_factory=dict)
    duplicate_count: int = 1

    @property
    def best_snippet(self) -> str:
        return max(self.snippets, key=len) if self.snippets else ""


class RankedSource(DeduplicatedSource):
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class FetchedDocument(BaseModel):
    source_id: str
    title: str = ""
    url: str
    domain: str = ""
    content: str = ""
    retrieved_at: str | None = None
    extraction_status: ExtractionStatus = ExtractionStatus.failed
    error: str | None = None
    content_chars: int = 0

    @property
    def usable(self) -> bool:
        return self.extraction_status in (ExtractionStatus.success, ExtractionStatus.partial)
