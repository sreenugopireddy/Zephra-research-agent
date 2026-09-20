"""Structured-output schema for the final synthesis step."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SynthesizedClaim(BaseModel):
    text: str = Field(description="A single factual claim made in the answer.")
    source_ids: list[str] = Field(
        default_factory=list, description="IDs of supplied sources supporting this claim."
    )
    confidence: Literal["high", "medium", "low"] = "medium"


class FinalAnswer(BaseModel):
    answer_markdown: str = Field(description="The full answer in markdown, with [S1] citations.")
    claims: list[SynthesizedClaim] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    source_ids_used: list[str] = Field(default_factory=list)
