"""Structured-output schemas used for the LLM reasoning steps."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QueryIntent(StrEnum):
    fact = "fact"
    explanation = "explanation"
    comparison = "comparison"
    how_to = "how_to"
    recommendation = "recommendation"
    research = "research"
    current_event = "current_event"
    multi_part = "multi_part"


class QueryAnalysis(BaseModel):
    intent: QueryIntent = Field(
        default=QueryIntent.research, description="The dominant intent of the user's question."
    )
    requires_current_information: bool = Field(
        default=False, description="True when the answer depends on recent or changing facts."
    )
    requires_multiple_queries: bool = Field(
        default=False, description="True when one search query cannot cover the question."
    )
    sub_questions: list[str] = Field(
        default_factory=list, description="Two to five answerable sub-questions."
    )
    preferred_source_types: list[str] = Field(
        default_factory=list,
        description="e.g. official_documentation, research_paper, government, technical_blog, news",
    )
    expected_answer_format: str = Field(
        default="prose", description="e.g. prose, comparison_table, step_by_step, short_answer"
    )


class SearchPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)
    provider_strategy: str = "parallel_all_providers"
    required_source_count: int = 3
    maximum_sources_to_fetch: int = 8


class EvidenceAssessment(BaseModel):
    sufficient: bool = Field(
        default=False, description="True when the evidence can answer the question."
    )
    covered_sub_questions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    needs_more_search: bool = False
    refined_queries: list[str] = Field(default_factory=list)
