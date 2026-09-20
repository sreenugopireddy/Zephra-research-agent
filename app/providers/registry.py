"""Builds the active provider set. Tests override this to inject fakes."""
from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.base import SearchProvider
from app.providers.serper_provider import SerperProvider
from app.providers.tavily_provider import TavilyProvider

_override: list[SearchProvider] | None = None


def build_providers(settings: Settings | None = None) -> list[SearchProvider]:
    settings = settings or get_settings()
    providers: list[SearchProvider] = [
        TavilyProvider(api_key=settings.tavily_api_key, timeout=settings.request_timeout_seconds),
        SerperProvider(api_key=settings.serper_api_key, timeout=settings.request_timeout_seconds),
    ]
    return providers


def get_providers(settings: Settings | None = None) -> list[SearchProvider]:
    if _override is not None:
        return _override
    return build_providers(settings)


def set_providers(providers: list[SearchProvider] | None) -> None:
    """Install (or clear) a provider override. Used by tests only."""
    global _override
    _override = providers
