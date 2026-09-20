"""Entry point that runs one research request through the graph."""
from __future__ import annotations

import uuid
from typing import Any

from app.graph.builder import get_compiled_graph
from app.graph.state import initial_state
from app.observability.logging import log_event
from app.schemas.requests import ResearchRequest
from app.schemas.responses import ResearchResponse


async def run_research(request: ResearchRequest, request_id: str | None = None) -> ResearchResponse:
    request_id = request_id or str(uuid.uuid4())
    options: dict[str, Any] = {
        "include_debug": request.options.include_debug,
    }
    if request.options.max_iterations is not None:
        options["max_iterations"] = request.options.max_iterations
    if request.options.max_sources is not None:
        options["max_sources"] = request.options.max_sources

    log_event("request_started", request_id=request_id, question_chars=len(request.question))
    state = initial_state(request_id, request.question, options)

    # recursion_limit is a second, independent guard against runaway looping
    final_state = await get_compiled_graph().ainvoke(state, config={"recursion_limit": 40})

    response = final_state.get("final_response")
    if response is None:  # defensive: the graph always ends at build_response
        return ResearchResponse(request_id=request_id, answer="", provider_status={})
    return response
