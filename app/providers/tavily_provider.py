"""Tavily search provider.

Calls the documented Tavily REST endpoint directly through httpx rather than
going through the langchain-tavily wrapper. Reason: we need a per-provider
timeout, an async call path, and precise transient/permanent error
classification for the retry policy - none of which the sync wrapper exposes
cleanly. The returned objects are plain `SearchResult` models, so the provider
stays swappable.
"""
from __future__ import annotations

import httpx

from app.providers.base import (
    PermanentProviderError,
    SearchProvider,
    TransientProviderError,
    classify_http_error,
)
from app.schemas.search import SearchResult

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(
        self,
        api_key: str = "",
        timeout: float = 15.0,
        max_results: int = 5,
        search_depth: str = "advanced",
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(timeout=timeout, max_results=max_results)
        self.api_key = api_key
        self.search_depth = search_depth
        self._client = client

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _payload(self, query: str) -> dict:
        return {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

    async def _search_once(self, query: str) -> list[SearchResult]:
        client = self._client or httpx.AsyncClient()
        owns_client = self._client is None
        try:
            response = await client.post(
                TAVILY_ENDPOINT, json=self._payload(query), timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise classify_http_error(exc, self.name) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientProviderError(
                f"tavily transport failure: {type(exc).__name__}", provider=self.name
            ) from exc
        except ValueError as exc:
            raise PermanentProviderError(
                "tavily returned a non-JSON body", provider=self.name
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

        return self._parse(data, query)

    def _parse(self, data: dict, query: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        for index, item in enumerate((data or {}).get("results") or []):
            url = (item or {}).get("url")
            if not url:
                continue
            raw_score = item.get("score")
            results.append(
                SearchResult(
                    title=item.get("title") or "",
                    url=url,
                    snippet=item.get("content") or "",
                    published_date=item.get("published_date"),
                    provider=self.name,
                    provider_rank=index,
                    provider_score=(
                        float(raw_score) if isinstance(raw_score, (int, float)) else None
                    ),
                    matched_query=query,
                )
            )
        return results
