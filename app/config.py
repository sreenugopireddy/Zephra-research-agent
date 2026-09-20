"""Application settings. All values come from the environment (see .env.example).

Credentials are never hardcoded and never logged.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- credentials -------------------------------------------------------
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    tavily_api_key: str = ""
    serper_api_key: str = ""
    google_api_key: str = ""
    google_cse_id: str = ""

    # --- bounds ------------------------------------------------------------
    request_timeout_seconds: float = 15.0
    max_search_iterations: int = Field(default=2, ge=1, le=2)
    max_queries_per_iteration: int = Field(default=6, ge=1, le=6)
    max_sources_to_fetch: int = Field(default=8, ge=1, le=8)
    max_sources_for_synthesis: int = Field(default=6, ge=1, le=6)

    # --- misc --------------------------------------------------------------
    log_level: str = "INFO"
    max_content_chars_per_source: int = 6000

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def serper_enabled(self) -> bool:
        return bool(self.serper_api_key)

    @property
    def google_enabled(self) -> bool:
        return bool(self.google_api_key and self.google_cse_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
