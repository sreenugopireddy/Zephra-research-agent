"""Serper.dev search provider — a Google-results proxy."""
from __future__ import annotations

import httpx

from app.providers.base import (
    PermanentProviderError,
    SearchProvider,
    TransientProviderError,
    classify_http_error,
)
from app.schemas.search import SearchResult

SERPER_ENDPOINT = "https://google.serper.dev/search"


class SerperProvider(SearchProvider):
    name = "serper"

    def __init__(
        self,
        api_key: str = "",
        timeout: float = 15.0,
        max_results: int = 5,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(timeout=timeout, max_results=max_results)
        self.api_key = api_key
        self._client = client

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _payload(self, query: str) -> dict:
        return {"q": query, "num": self.max_results}

    async def _search_once(self, query: str) -> list[SearchResult]:
        client = self._client or httpx.AsyncClient()
        owns_client = self._client is None
        try:
            response = await client.post(
                SERPER_ENDPOINT,
                json=self._payload(query),
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise classify_http_error(exc, self.name) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientProviderError(
                f"serper transport failure: {type(exc).__name__}", provider=self.name
            ) from exc
        except ValueError as exc:
            raise PermanentProviderError(
                "serper returned a non-JSON body", provider=self.name
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

        return self._parse(data, query)

    def _parse(self, data: dict, query: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in (data or {}).get("organic") or []:
            url = (item or {}).get("link")
            if not url:
                continue
            position = item.get("position")
            results.append(
                SearchResult(
                    title=item.get("title") or "",
                    url=url,
                    snippet=item.get("snippet") or "",
                    published_date=item.get("date"),
                    provider=self.name,
                    provider_rank=(position - 1) if isinstance(position, int) else len(results),
                    provider_score=None,
                    matched_query=query,
                )
            )
        return results