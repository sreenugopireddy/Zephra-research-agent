"""FastAPI application exposing the research agent."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.graph.builder import get_compiled_graph
from app.graph.runner import run_research
from app.observability.logging import configure_logging, log_event
from app.schemas.requests import ResearchRequest
from app.schemas.responses import ResearchResponse
from fastapi.responses import HTMLResponse, JSONResponse
from app.dashboard import DASHBOARD_HTML

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    get_compiled_graph()  # compile once at startup
    log_event(
        "startup",
        tavily_configured=settings.tavily_enabled,
                serper_configured=settings.serper_enabled,
        groq_configured=bool(settings.groq_api_key),
        model=settings.groq_model,
    )
    yield
    log_event("shutdown")


app = FastAPI(
    title="Multi-Source Web Research Agent",
    version="0.1.0",
    description=(
        "Phase 1: a bounded LangGraph workflow that searches multiple independent "
        "web-search providers, deduplicates and ranks sources, and produces an "
        "evidence-grounded, citation-validated answer."
    ),
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
            "providers_configured": {
            "tavily": settings.tavily_enabled,
            "serper": settings.serper_enabled,
        },
        "groq_configured": bool(settings.groq_api_key),
    }
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD_HTML

@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest, http_request: Request) -> ResearchResponse:
    request_id = http_request.headers.get("x-request-id") or str(uuid.uuid4())
    try:
        return await run_research(request, request_id=request_id)
    except Exception as exc:
        log_event(
            "request_error", level="error", request_id=request_id, error=type(exc).__name__
        )
        raise HTTPException(
            status_code=500, detail=f"research failed: {type(exc).__name__}"
        ) from exc


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
