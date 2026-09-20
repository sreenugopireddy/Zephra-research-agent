"""Google Programmable Search (Custom Search JSON API) provider.

A small LangChain-compatible provider written against the documented REST
endpoint so it can be run asynchronously with its own timeout and retry policy.
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

GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchProvider(SearchProvider):
    name = "google"

    def __init__(
        self,
        api_key: str = "",
        cse_id: str = "",
        timeout: float = 15.0,
        max_results: int = 5,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(timeout=timeout, max_results=max_results)
        self.api_key = api_key
        self.cse_id = cse_id
        self._client = client

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.cse_id)

    def _params(self, query: str) -> dict:
        return {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(self.max_results, 10),
            "safe": "active",
        }

    async def _search_once(self, query: str) -> list[SearchResult]:
        client = self._client or httpx.AsyncClient()
        owns_client = self._client is None
        try:
            response = await client.get(
                GOOGLE_ENDPOINT, params=self._params(query), timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise classify_http_error(exc, self.name) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientProviderError(
                f"google transport failure: {type(exc).__name__}", provider=self.name
            ) from exc
        except ValueError as exc:
            raise PermanentProviderError(
                "google returned a non-JSON body", provider=self.name
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

        return self._parse(data, query)

    @staticmethod
    def _published_date(item: dict) -> str | None:
        pagemap = item.get("pagemap") or {}
        for entry in (pagemap.get("metatags") or []):
            for key in ("article:published_time", "datepublished", "og:updated_time"):
                if entry.get(key):
                    return entry[key]
        return None

    def _parse(self, data: dict, query: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        for index, item in enumerate((data or {}).get("items") or []):
            url = (item or {}).get("link")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=item.get("title") or "",
                    url=url,
                    snippet=item.get("snippet") or "",
                    published_date=self._published_date(item),
                    provider=self.name,
                    provider_rank=index,
                    provider_score=None,  # Google does not expose a relevance score
                    matched_query=query,
                )
            )
        return results
