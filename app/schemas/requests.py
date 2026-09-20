"""Inbound API schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ResearchOptions(BaseModel):
    max_iterations: int | None = Field(default=None, ge=1, le=2)
    max_sources: int | None = Field(default=None, ge=1, le=6)
    include_debug: bool = False


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    options: ResearchOptions = Field(default_factory=ResearchOptions)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned
