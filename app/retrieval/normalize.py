"""Convert provider-specific payloads into the common NormalizedResult schema."""
from __future__ import annotations

import html
import re
from collections.abc import Iterable

from app.retrieval.canonicalize import canonicalize_url, domain_of
from app.schemas.search import NormalizedResult, SearchResult

_WS = re.compile(r"\s+")
_TAGS = re.compile(r"<[^>]+>")


def clean_text(value: str | None, limit: int = 1200) -> str:
    """Unescape entities, drop stray markup and collapse whitespace."""
    if not value:
        return ""
    text = _WS.sub(" ", html.unescape(_TAGS.sub(" ", str(value)))).strip()
    return text[:limit]


def normalize_result(result: SearchResult) -> NormalizedResult | None:
    """Attach canonical URL + domain. Returns None for unusable rows."""
    canonical = canonicalize_url(result.url)
    if not canonical:
        return None
    return NormalizedResult(
        title=clean_text(result.title, 400) or domain_of(canonical),
        url=result.url.strip(),
        snippet=clean_text(result.snippet),
        published_date=result.published_date,
        provider=result.provider,
        provider_rank=result.provider_rank,
        provider_score=result.provider_score,
        matched_query=result.matched_query,
        redirect_target=result.redirect_target,
        canonical_url=canonical,
        domain=domain_of(canonical),
    )


def normalize_results(results: Iterable[SearchResult]) -> list[NormalizedResult]:
    """Normalize a batch, silently skipping rows without a usable URL."""
    normalized: list[NormalizedResult] = []
    for result in results:
        item = normalize_result(result)
        if item is not None:
            normalized.append(item)
    return normalized
