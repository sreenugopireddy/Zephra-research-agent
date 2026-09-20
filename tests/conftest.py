"""Shared fixtures. No test ever needs a real API key."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("TAVILY_API_KEY", "")
os.environ.setdefault("GOOGLE_API_KEY", "")

from app.config import get_settings  # noqa: E402
from app.llm.groq_client import set_llm_client  # noqa: E402
from app.providers.registry import set_providers  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_overrides():
    get_settings.cache_clear()
    set_providers(None)
    set_llm_client(None)
    yield
    set_providers(None)
    set_llm_client(None)
    get_settings.cache_clear()
