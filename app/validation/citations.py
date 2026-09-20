"""Deterministic citation validation.

The LLM is never trusted to police its own citations. This module checks, for a
synthesized answer:
  * every inline [S#] marker in the prose refers to a supplied source;
  * every claim.source_ids entry refers to a supplied source;
  * every cited source was actually selected AND carries a valid http(s) URL;
  * no unknown citation IDs appear anywhere.

Sources whose extraction failed are "not fully inspected": citing them is not an
error, but claims resting only on them are downgraded rather than dropped.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

from app.llm.structured_models import FinalAnswer
from app.schemas.responses import Claim
from app.schemas.search import ExtractionStatus, FetchedDocument
from app.validation.safety import is_safe_url

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


class CitationValidation(BaseModel):
    valid: bool = True
    cited_ids: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list)
    uncited_claims: list[str] = Field(default_factory=list)
    invalid_url_ids: list[str] = Field(default_factory=list)
    not_fully_inspected_ids: list[str] = Field(default_factory=list)
    reason: str = "ok"


def extract_citation_ids(text: str) -> list[str]:
    """Every [S#] marker in the given markdown, in order of first appearance."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(text or ""):
        if match not in seen:
            seen.append(match)
    return seen


def validate_citations(
    answer: FinalAnswer, documents: Sequence[FetchedDocument]
) -> CitationValidation:
    """Check an answer against the evidence that was actually supplied."""
    valid_ids = {doc.source_id for doc in documents}
    url_ok = {doc.source_id for doc in documents if is_safe_url(doc.url)}
    inspected = {
        doc.source_id
        for doc in documents
        if doc.extraction_status == ExtractionStatus.success
    }

    cited: list[str] = extract_citation_ids(answer.answer_markdown)
    for claim in answer.claims:
        for sid in claim.source_ids:
            if sid not in cited:
                cited.append(sid)
    for sid in answer.source_ids_used:
        if sid not in cited:
            cited.append(sid)

    unknown = [sid for sid in cited if sid not in valid_ids]
    invalid_url = [sid for sid in cited if sid in valid_ids and sid not in url_ok]
    partial_only = [sid for sid in cited if sid in valid_ids and sid not in inspected]
    uncited = [c.text for c in answer.claims if not c.source_ids]

    problems: list[str] = []
    if unknown:
        problems.append(f"unknown source ids: {', '.join(unknown)}")
    if invalid_url:
        problems.append(f"cited sources without a valid url: {', '.join(invalid_url)}")
    if not cited and documents:
        problems.append("answer contains no citations despite available evidence")

    return CitationValidation(
        valid=not problems,
        cited_ids=cited,
        unknown_ids=unknown,
        uncited_claims=uncited,
        invalid_url_ids=invalid_url,
        not_fully_inspected_ids=partial_only,
        reason="; ".join(problems) if problems else "ok",
    )


def sanitize_answer(
    answer: FinalAnswer, validation: CitationValidation
) -> tuple[str, list[Claim]]:
    """Produce a safe response when validation still fails after the retry.

    Unsupported claims are kept but marked `unsupported` with their bad IDs
    removed, so nothing is silently fabricated and nothing is silently dropped.
    """
    bad = set(validation.unknown_ids) | set(validation.invalid_url_ids)
    weak = set(validation.not_fully_inspected_ids)

    cleaned: list[Claim] = []
    for claim in answer.claims:
        kept = [sid for sid in claim.source_ids if sid not in bad]
        if not kept:
            confidence = "unsupported"
        elif any(sid in weak for sid in kept) and claim.confidence == "high":
            confidence = "medium"
        else:
            confidence = claim.confidence
        cleaned.append(Claim(text=claim.text, source_ids=kept, confidence=confidence))

    text = answer.answer_markdown or ""
    for sid in bad:
        text = text.replace(f"[{sid}]", "[unsupported]")
    if bad:
        text += (
            "\n\n> **Note:** some citations produced for this answer could not be "
            "verified against the retrieved sources and have been marked unsupported."
        )
    return text, cleaned


def to_response_claims(claims: Iterable) -> list[Claim]:
    return [
        Claim(text=c.text, source_ids=list(c.source_ids), confidence=c.confidence) for c in claims
    ]
