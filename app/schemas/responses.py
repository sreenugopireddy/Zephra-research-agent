"""Outbound API schemas."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low", "unsupported"]


class Claim(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = "medium"


class SourceRef(BaseModel):
    source_id: str
    title: str = ""
    url: str
    provider: str = ""
    domain: str = ""
    extraction_status: str = "failed"


class ResearchMetadata(BaseModel):
    search_iterations: int = 0
    sources_considered: int = 0
    sources_selected: int = 0
    queries_executed: int = 0
    duplicates_merged: int = 0
    citation_validation: str = "not_run"
    stop_reason: str = "completed"
    elapsed_seconds: float = 0.0


class ResearchResponse(BaseModel):
    request_id: str
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    provider_status: dict[str, str] = Field(default_factory=dict)
    metadata: ResearchMetadata = Field(default_factory=ResearchMetadata)
    debug: dict[str, Any] | None = None
