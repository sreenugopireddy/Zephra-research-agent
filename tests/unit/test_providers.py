"""Provider behaviour, exercised with httpx MockTransport - no network, no keys."""
from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.providers.base import (
    PermanentProviderError,
    TransientProviderError,
    classify_http_error,
)
from app.providers.google_provider import GoogleSearchProvider
from app.providers.registry import build_providers, get_providers, set_providers
from app.providers.tavily_provider import TavilyProvider


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_tavily_parses_and_normalizes_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://a.com/x",
                        "title": "A",
                        "content": "snippet",
                        "score": 0.77,
                        "published_date": "2025-01-01",
                    }
                ]
            },
        )

    async with client_for(handler) as client:
        provider = TavilyProvider(api_key="k", client=client)
        results = await provider.search("q")

    assert len(results) == 1
    assert results[0].provider == "tavily"
    assert results[0].provider_score == 0.77
    assert results[0].matched_query == "q"
    assert results[0].provider_rank == 0


async def test_google_parses_payload_and_reads_published_date():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "link": "https://b.com/y",
                        "title": "B",
                        "snippet": "s",
                        "pagemap": {
                            "metatags": [{"article:published_time": "2025-03-02T00:00:00Z"}]
                        },
                    }
                ]
            },
        )

    async with client_for(handler) as client:
        provider = GoogleSearchProvider(api_key="k", cse_id="c", client=client)
        results = await provider.search("q")

    assert results[0].provider == "google"
    assert results[0].published_date == "2025-03-02T00:00:00Z"


async def test_timeout_is_retried_then_raises_transient():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ReadTimeout("timed out", request=request)

    async with client_for(handler) as client:
        provider = TavilyProvider(api_key="k", client=client)
        provider.max_attempts = 3
        with pytest.raises(TransientProviderError):
            await provider.search("q")

    assert attempts["n"] == 3, "transient failures must be retried up to max_attempts"


async def test_per_provider_timeout_is_passed_to_the_request():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"results": []})

    async with client_for(handler) as client:
        await TavilyProvider(api_key="k", timeout=3.5, client=client).search("q")

    assert seen["timeout"]["read"] == 3.5


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_retryable_statuses_are_transient(status):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={})

    async with client_for(handler) as client:
        provider = TavilyProvider(api_key="k", client=client)
        provider.max_attempts = 2
        with pytest.raises(TransientProviderError):
            await provider.search("q")
    assert calls["n"] == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_client_errors_are_permanent_and_not_retried(status):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={})

    async with client_for(handler) as client:
        provider = GoogleSearchProvider(api_key="bad", cse_id="c", client=client)
        with pytest.raises(PermanentProviderError):
            await provider.search("q")
    assert calls["n"] == 1, "invalid credentials / malformed requests must not be retried"


async def test_unconfigured_provider_raises_permanently():
    with pytest.raises(PermanentProviderError):
        await TavilyProvider(api_key="").search("q")
    assert TavilyProvider(api_key="").enabled is False
    assert GoogleSearchProvider(api_key="k", cse_id="").enabled is False


def test_classify_http_error_taxonomy():
    request = httpx.Request("GET", "https://x.com")
    transient = httpx.HTTPStatusError("", request=request, response=httpx.Response(503))
    permanent = httpx.HTTPStatusError("", request=request, response=httpx.Response(401))
    assert isinstance(classify_http_error(transient, "tavily"), TransientProviderError)
    assert isinstance(classify_http_error(permanent, "tavily"), PermanentProviderError)


def test_registry_builds_both_providers_and_honours_overrides():
    providers = build_providers(Settings(tavily_api_key="a", serper_api_key="b"))
    assert [p.name for p in providers] == ["tavily", "serper"]
    set_providers([])
    assert get_providers() == []
    set_providers(None)
