"""Common search-provider interface and error taxonomy.

Errors are split into two families so retries are never wasted:

TransientProviderError  timeouts, 408, 429, 5xx, connection resets -> retried
PermanentProviderError  401/403 (bad credentials), 400/404 (malformed request),
                        missing configuration -> never retried
"""
from __future__ import annotations

import abc

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.schemas.search import SearchResult

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class ProviderError(Exception):
    """Base class for provider failures."""

    def __init__(self, message: str, *, provider: str = "", status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class TransientProviderError(ProviderError):
    """Temporary failure; retrying may succeed."""


class PermanentProviderError(ProviderError):
    """Permanent failure such as invalid credentials or a malformed request."""


def classify_http_error(exc: httpx.HTTPStatusError, provider: str) -> ProviderError:
    """Map an HTTP status onto the transient/permanent taxonomy."""
    status = exc.response.status_code
    if status in RETRYABLE_STATUS_CODES:
        return TransientProviderError(
            f"{provider} returned retryable status {status}", provider=provider, status_code=status
        )
    if status in (401, 403):
        return PermanentProviderError(
            f"{provider} rejected the credentials (status {status})",
            provider=provider,
            status_code=status,
        )
    return PermanentProviderError(
        f"{provider} returned non-retryable status {status}", provider=provider, status_code=status
    )


class SearchProvider(abc.ABC):
    """Interface every web-search provider implements."""

    name: str = "provider"
    max_attempts: int = 3

    def __init__(self, timeout: float = 15.0, max_results: int = 5):
        self.timeout = timeout
        self.max_results = max_results

    @property
    def enabled(self) -> bool:
        """False when the provider lacks the configuration it needs."""
        return True

    @abc.abstractmethod
    async def _search_once(self, query: str) -> list[SearchResult]:
        """Perform a single search call. Raises ProviderError subclasses."""

    async def search(self, query: str) -> list[SearchResult]:
        """Search with bounded retries on transient failures only."""
        if not self.enabled:
            raise PermanentProviderError(
                f"{self.name} is not configured", provider=self.name
            )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=0.4, min=0.4, max=3),
            retry=retry_if_exception_type(TransientProviderError),
            reraise=True,
        ):
            with attempt:
                return await self._search_once(query)
        return []  # pragma: no cover - AsyncRetrying always returns or raises
