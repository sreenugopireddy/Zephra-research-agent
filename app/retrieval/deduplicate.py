"""Merge results that point at the same real-world source.

Matching keys, in priority order:
  1. Canonical URL.
  2. Canonical redirect target, when a provider supplied one.
  3. Normalized title + domain (catches the same article on two URLs).

Two providers returning the same URL is ONE source, not two. Provenance from
every copy is preserved: found_by, matched_queries, ranks, scores, snippets.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from app.retrieval.canonicalize import canonicalize_url
from app.schemas.search import DeduplicatedSource, NormalizedResult

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _title_key(title: str, domain: str) -> str | None:
    slug = _NON_WORD.sub(" ", (title or "").lower()).strip()
    if len(slug) < 15:  # too short to be a reliable identity signal
        return None
    return f"{domain}::{slug}"


def _merge_into(source: DeduplicatedSource, result: NormalizedResult) -> None:
    source.duplicate_count += 1
    if result.provider not in source.found_by:
        source.found_by.append(result.provider)
    if result.matched_query and result.matched_query not in source.matched_queries:
        source.matched_queries.append(result.matched_query)
    if result.snippet and result.snippet not in source.snippets:
        source.snippets.append(result.snippet)
    existing_rank = source.provider_ranks.get(result.provider)
    if existing_rank is None or result.provider_rank < existing_rank:
        source.provider_ranks[result.provider] = result.provider_rank
    if result.provider_score is not None:
        source.provider_scores[result.provider] = max(
            result.provider_score, source.provider_scores.get(result.provider, 0.0)
        )
    if not source.published_date and result.published_date:
        source.published_date = result.published_date
    if len(result.title) > len(source.title):
        source.title = result.title


def deduplicate_results(
    results: Iterable[NormalizedResult], start_index: int = 1
) -> list[DeduplicatedSource]:
    """Collapse duplicates and assign stable S1..Sn identifiers."""
    by_key: dict[str, DeduplicatedSource] = {}
    ordered: list[DeduplicatedSource] = []
    next_index = start_index

    for result in results:
        keys = [result.canonical_url]
        if result.redirect_target:
            redirect = canonicalize_url(result.redirect_target)
            if redirect:
                keys.append(redirect)
        title_key = _title_key(result.title, result.domain)
        if title_key:
            keys.append(title_key)

        existing = next((by_key[k] for k in keys if k in by_key), None)
        if existing is not None:
            _merge_into(existing, result)
            for key in keys:
                by_key.setdefault(key, existing)
            continue

        source = DeduplicatedSource(
            source_id=f"S{next_index}",
            title=result.title,
            url=result.url,
            canonical_url=result.canonical_url,
            domain=result.domain,
            snippets=[result.snippet] if result.snippet else [],
            published_date=result.published_date,
            found_by=[result.provider],
            matched_queries=[result.matched_query] if result.matched_query else [],
            provider_ranks={result.provider: result.provider_rank},
            provider_scores=(
                {result.provider: result.provider_score}
                if result.provider_score is not None
                else {}
            ),
            duplicate_count=1,
        )
        next_index += 1
        for key in keys:
            by_key[key] = source
        ordered.append(source)

    return ordered
