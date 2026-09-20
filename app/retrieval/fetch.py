"""Controlled source fetching and text extraction.

Every URL is safety-checked before a request is made. Each page is fetched
independently: one failure never aborts the batch. Content is truncated before
it can reach the LLM.

Extraction status
-----------------
success  full page text extracted (>= _MIN_SUCCESS_CHARS characters)
partial  page unavailable or too thin, so only the search snippet is retained
failed   nothing usable; the source must not be cited as fully inspected
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from html.parser import HTMLParser

import httpx

from app.observability.logging import log_event
from app.schemas.search import ExtractionStatus, FetchedDocument, RankedSource
from app.validation.safety import is_safe_url, reject_reason

_MIN_SUCCESS_CHARS = 400
_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"}
_USER_AGENT = "ZephraResearchAgent/0.1 (+phase1; respectful crawler)"


class _TextExtractor(HTMLParser):
    """Minimal readability pass over HTML using only the standard library."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title = (self.title + " " + text).strip()
        else:
            self._chunks.append(text)

    def text(self) -> str:
        joined = " ".join(self._chunks)
        lines = [" ".join(line.split()) for line in joined.split("\n")]
        return "\n".join(line for line in lines if line)


def extract_text(html: str) -> tuple[str, str]:
    """Return (text, document_title) for an HTML string. Never raises."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: keep whatever was parsed
        pass
    return parser.text(), parser.title


def _snippet_document(source: RankedSource, error: str) -> FetchedDocument:
    """Fall back to the provider snippet as partial evidence."""
    snippet = source.best_snippet
    return FetchedDocument(
        source_id=source.source_id,
        title=source.title,
        url=source.url,
        domain=source.domain,
        content=snippet,
        retrieved_at=datetime.now(UTC).isoformat(),
        extraction_status=ExtractionStatus.partial if snippet else ExtractionStatus.failed,
        error=error,
        content_chars=len(snippet),
    )


async def fetch_source(
    client: httpx.AsyncClient, source: RankedSource, max_chars: int, timeout: float
) -> FetchedDocument:
    """Fetch and extract one source. Failures degrade, they do not raise."""
    if not is_safe_url(source.url):
        reason = reject_reason(source.url)
        log_event("fetch_rejected", level="warning", source_id=source.source_id, reason=reason)
        return _snippet_document(source, f"rejected: {reason}")

    try:
        response = await client.get(
            source.url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
        )
        response.raise_for_status()
    except Exception as exc:
        log_event(
            "fetch_failed",
            level="warning",
            source_id=source.source_id,
            domain=source.domain,
            error=type(exc).__name__,
        )
        return _snippet_document(source, f"{type(exc).__name__}: {exc}")

    content_type = response.headers.get("content-type", "")
    if "html" in content_type or "text" in content_type:
        text, doc_title = extract_text(response.text)
    else:
        return _snippet_document(source, f"unsupported content-type '{content_type}'")

    if len(text) < _MIN_SUCCESS_CHARS:
        merged = (text + "\n" + source.best_snippet).strip()
        return FetchedDocument(
            source_id=source.source_id,
            title=source.title or doc_title,
            url=str(response.url),
            domain=source.domain,
            content=merged[:max_chars],
            retrieved_at=datetime.now(UTC).isoformat(),
            extraction_status=(
                ExtractionStatus.partial if merged else ExtractionStatus.failed
            ),
            error="extracted content below usable threshold" if not merged else None,
            content_chars=len(merged[:max_chars]),
        )

    truncated = text[:max_chars]
    return FetchedDocument(
        source_id=source.source_id,
        title=source.title or doc_title,
        url=str(response.url),
        domain=source.domain,
        content=truncated,
        retrieved_at=datetime.now(UTC).isoformat(),
        extraction_status=ExtractionStatus.success,
        content_chars=len(truncated),
    )


async def fetch_documents(
    sources: Sequence[RankedSource], max_chars: int = 6000, timeout: float = 15.0
) -> list[FetchedDocument]:
    """Fetch the given sources concurrently, preserving source IDs and order."""
    if not sources:
        return []
    async with httpx.AsyncClient() as client:
        tasks = [fetch_source(client, s, max_chars, timeout) for s in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    documents: list[FetchedDocument] = []
    for source, result in zip(sources, results, strict=False):
        if isinstance(result, BaseException):
            documents.append(_snippet_document(source, f"unexpected: {result}"))
        else:
            documents.append(result)
    return documents
