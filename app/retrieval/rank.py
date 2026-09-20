"""Transparent, deterministic source ranking.

    score = 0.35*relevance
          + 0.20*authority
          + 0.15*freshness
          + 0.10*content_availability
          + 0.20*provider_agreement

Every component is in [0, 1], so the final score is in [0, 1]. The breakdown is
stored on each ranked source so a reviewer can see exactly why it placed where
it did. The LLM never replaces this policy; it only consumes the ordered output.

Components
----------
relevance            Jaccard-style token overlap between the question (plus the
                     queries that matched) and the title+snippet, blended with a
                     positional decay over the provider's own rank and its score.
authority            Domain class: .gov/.edu and known standards bodies rank
                     highest, then official docs and research repositories, then
                     reputable technical publications, then everything else.
freshness            Recency of published_date. Only weighted when the query
                     needs current information; otherwise a neutral 0.5.
content_availability Proxy for how much text we already hold (snippet length).
provider_agreement   1.0 when independent providers both surfaced the source.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from app.schemas.search import DeduplicatedSource, RankedSource

WEIGHTS = {
    "relevance": 0.35,
    "authority": 0.20,
    "freshness": 0.15,
    "content_availability": 0.10,
    "provider_agreement": 0.20,
}

HIGH_AUTHORITY_SUFFIXES = (".gov", ".gov.uk", ".edu", ".int", ".mil", ".ac.uk")
HIGH_AUTHORITY_DOMAINS = {
    "arxiv.org", "nature.com", "science.org", "ieee.org", "acm.org", "nih.gov",
    "who.int", "worldbank.org", "oecd.org", "nist.gov", "rfc-editor.org", "ietf.org",
    "w3.org", "iso.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
}
OFFICIAL_DOC_DOMAINS = {
    "docs.python.org", "developer.mozilla.org", "kubernetes.io", "docs.aws.amazon.com",
    "cloud.google.com", "learn.microsoft.com", "docs.microsoft.com", "python.langchain.com",
    "docs.langchain.com", "platform.openai.com", "docs.anthropic.com", "pytorch.org",
    "tensorflow.org", "postgresql.org", "redis.io", "fastapi.tiangolo.com", "openai.com",
}
REPUTABLE_TECH_DOMAINS = {
    "github.com", "stackoverflow.com", "huggingface.co", "arstechnica.com", "reuters.com",
    "apnews.com", "bbc.co.uk", "nytimes.com", "economist.com", "wired.com", "acm.queue.org",
    "martinfowler.com", "databricks.com", "cloudflare.com", "wikipedia.org",
}
LOW_QUALITY_DOMAINS = {
    "pinterest.com", "quora.com", "answers.com", "ehow.com", "blogspot.com",
    "medium.com", "facebook.com", "x.com", "twitter.com", "reddit.com",
}

_STOPWORDS = frozenset(
    """a an and are as at be but by for from how if in into is it its of on or that the
    their there these this to was what when where which who why with your you""".split()
)
_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 2}


def score_relevance(source: DeduplicatedSource, question: str) -> float:
    """Token overlap with the question, blended with provider rank/score."""
    wanted = _tokens(question) | _tokens(" ".join(source.matched_queries))
    have = _tokens(source.title) | _tokens(source.best_snippet)
    overlap = len(wanted & have) / len(wanted) if wanted else 0.0

    best_rank = min(source.provider_ranks.values()) if source.provider_ranks else 9
    rank_component = 1.0 / (1.0 + 0.35 * best_rank)
    provider_score = max(source.provider_scores.values(), default=0.0)

    return min(1.0, 0.55 * overlap + 0.30 * rank_component + 0.15 * min(provider_score, 1.0))


def score_authority(domain: str, preferred_source_types: Sequence[str] = ()) -> float:
    """Classify the domain into a small, auditable set of authority tiers."""
    domain = (domain or "").lower()
    if not domain:
        return 0.3
    base = 0.45
    if domain in LOW_QUALITY_DOMAINS or any(domain.endswith("." + d) for d in LOW_QUALITY_DOMAINS):
        base = 0.20
    elif domain.endswith(HIGH_AUTHORITY_SUFFIXES) or domain in HIGH_AUTHORITY_DOMAINS:
        base = 1.00
    elif domain in OFFICIAL_DOC_DOMAINS or domain.startswith("docs."):
        base = 0.90
    elif domain in REPUTABLE_TECH_DOMAINS or domain.endswith(".org"):
        base = 0.70

    wants_official = any(
        t in {"official_documentation", "research_paper", "government", "standards"}
        for t in preferred_source_types
    )
    if wants_official and base >= 0.90:
        base = min(1.0, base + 0.05)
    return base


def score_freshness(published_date: str | None, freshness_required: bool) -> float:
    """Recency score; neutral (0.5) when the question is not time-sensitive."""
    if not freshness_required:
        return 0.5
    if not published_date:
        return 0.35
    parsed: datetime | None = None
    text = str(published_date).strip().replace("Z", "+00:00")
    for candidate in (text, text[:10]):
        try:
            parsed = datetime.fromisoformat(candidate)
            break
        except ValueError:
            continue
    if parsed is None:
        return 0.35
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    age_days = (datetime.now(UTC) - parsed).days
    if age_days < 0:
        return 0.8
    for cutoff, value in ((30, 1.0), (180, 0.85), (365, 0.65), (1095, 0.45)):
        if age_days <= cutoff:
            return value
    return 0.25


def score_content_availability(source: DeduplicatedSource) -> float:
    length = len(source.best_snippet)
    if length >= 400:
        return 1.0
    if length >= 200:
        return 0.8
    if length >= 80:
        return 0.6
    return 0.3 if length else 0.1


def score_provider_agreement(source: DeduplicatedSource) -> float:
    distinct = len(set(source.found_by))
    if distinct >= 2:
        return 1.0
    return 0.6 if source.duplicate_count > 1 else 0.4


def rank_sources(
    sources: Iterable[DeduplicatedSource],
    question: str,
    freshness_required: bool = False,
    preferred_source_types: Sequence[str] = (),
) -> list[RankedSource]:
    """Score and order sources, highest first. Ties break on source_id."""
    ranked: list[RankedSource] = []
    for source in sources:
        breakdown = {
            "relevance": round(score_relevance(source, question), 4),
            "authority": round(score_authority(source.domain, preferred_source_types), 4),
            "freshness": round(score_freshness(source.published_date, freshness_required), 4),
            "content_availability": round(score_content_availability(source), 4),
            "provider_agreement": round(score_provider_agreement(source), 4),
        }
        total = sum(WEIGHTS[name] * value for name, value in breakdown.items())
        ranked.append(
            RankedSource(
                **source.model_dump(), score=round(total, 4), score_breakdown=breakdown
            )
        )
    ranked.sort(key=lambda s: (-s.score, int(s.source_id[1:])))
    return ranked
