"""ChatGroq wrapper providing structured output for the three reasoning steps.

The client is created lazily so the application (and the test suite) can be
imported without any credentials present. `set_llm_client` lets tests inject a
fake implementation without patching LangChain internals.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.config import Settings, get_settings
from app.llm.prompts import (
    EVIDENCE_ASSESSMENT_SYSTEM,
    QUERY_ANALYSIS_SYSTEM,
    STRICT_RETRY_SUFFIX,
    SYNTHESIS_SYSTEM,
    evidence_block,
)
from app.llm.structured_models import FinalAnswer
from app.observability.logging import log_event
from app.schemas.evidence import EvidenceAssessment, QueryAnalysis
from app.schemas.search import FetchedDocument


class LLMUnavailableError(RuntimeError):
    """Raised when Groq cannot be reached or returns unusable output."""


class GroqLLMClient:
    """Thin, testable wrapper over ChatGroq structured output."""

    def __init__(self, settings: Settings | None = None, model: Any = None):
        self._settings = settings or get_settings()
        self._model = model

    # -- plumbing ---------------------------------------------------------
    def _chat_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self._settings.groq_api_key:
            raise LLMUnavailableError("GROQ_API_KEY is not configured")
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailableError("langchain-groq is not installed") from exc
        self._model = ChatGroq(
            model=self._settings.groq_model,
            api_key=self._settings.groq_api_key,
            temperature=0,
            timeout=self._settings.request_timeout_seconds,
            max_retries=2,
        )
        return self._model

    async def _structured(self, schema: type, system: str, user: str) -> Any:
        model = self._chat_model()
        schema_json = schema.model_json_schema()
        json_instructions = (
            "\n\nRespond with ONLY a single valid JSON object — no prose, no markdown "
            "code fences, no commentary before or after — matching this JSON schema "
            f"exactly:\n{schema_json}"
        )
        try:
            # json_mode instead of the default forced tool-calling: tool-calling
            # compliance varies across Groq-hosted models (gpt-oss family
            # sometimes answers in plain text instead of calling the tool,
            # raising groq.BadRequestError: tool_use_failed). json_mode is more
            # broadly supported and just needs the schema spelled out above.
            structured = model.with_structured_output(schema, method="json_mode")
            result = await structured.ainvoke(
                [
                    {"role": "system", "content": system + json_instructions},
                    {"role": "user", "content": user},
                ]
            )
        except Exception as exc:
            raise LLMUnavailableError(f"groq structured call failed: {type(exc).__name__}") from exc
        if result is None:
            raise LLMUnavailableError("groq returned an empty structured response")
        if isinstance(result, dict):
            return schema(**result)
        return result

    # -- reasoning steps --------------------------------------------------
    async def analyze_query(self, question: str) -> QueryAnalysis:
        return await self._structured(
            QueryAnalysis, QUERY_ANALYSIS_SYSTEM, f"Research question:\n{question}"
        )

    async def assess_evidence(
        self,
        question: str,
        sub_questions: Sequence[str],
        documents: Sequence[FetchedDocument],
        iteration: int,
        max_iterations: int,
    ) -> EvidenceAssessment:
        user = (
            f"Research question:\n{question}\n\n"
            f"Sub-questions:\n" + "\n".join(f"- {s}" for s in sub_questions) + "\n\n"
            f"Search iteration {iteration} of a maximum of {max_iterations}.\n\n"
            f"Retrieved evidence:\n{evidence_block(documents)}"
        )
        return await self._structured(EvidenceAssessment, EVIDENCE_ASSESSMENT_SYSTEM, user)

    async def synthesize(
        self,
        question: str,
        documents: Sequence[FetchedDocument],
        known_conflicts: Sequence[str] = (),
        missing_information: Sequence[str] = (),
        invalid_ids: Sequence[str] = (),
    ) -> FinalAnswer:
        valid_ids = [doc.source_id for doc in documents]
        system = SYNTHESIS_SYSTEM
        if invalid_ids:
            system += STRICT_RETRY_SUFFIX.format(
                invalid_ids=", ".join(invalid_ids) or "(none)",
                valid_ids=", ".join(valid_ids) or "(none)",
            )
        user = (
            f"Research question:\n{question}\n\n"
            f"Permitted source IDs: {', '.join(valid_ids) or '(none)'}\n\n"
            f"Known conflicts: {'; '.join(known_conflicts) or '(none reported)'}\n"
            f"Known evidence gaps: {'; '.join(missing_information) or '(none reported)'}\n\n"
            f"Evidence:\n{evidence_block(documents)}"
        )
        log_event("synthesis_requested", sources=len(documents), strict_retry=bool(invalid_ids))
        return await self._structured(FinalAnswer, system, user)


_client: GroqLLMClient | None = None


def get_llm_client() -> GroqLLMClient:
    global _client
    if _client is None:
        _client = GroqLLMClient()
    return _client


def set_llm_client(client: Any | None) -> None:
    """Install (or clear) an LLM override. Used by tests only."""
    global _client
    _client = client
