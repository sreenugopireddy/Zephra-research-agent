"""Assembles the bounded LangGraph workflow.

START -> validate_request -> analyze_query -> plan_search -> search_providers
      -> normalize_results -> deduplicate_results -> rank_sources -> fetch_sources
      -> assess_evidence -> (refine_search -> search_providers | synthesize_answer)
      -> validate_citations -> build_response -> END

This is an explicit graph, not an open-ended agent loop: the only cycle is
refine_search -> search_providers, and it is capped by route_after_evidence.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routes import route_after_evidence, route_after_validation
from app.graph.state import ResearchState


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("validate_request", nodes.validate_request)
    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("plan_search", nodes.plan_search)
    graph.add_node("search_providers", nodes.search_providers)
    graph.add_node("normalize_results", nodes.normalize_results_node)
    graph.add_node("deduplicate_results", nodes.deduplicate_results_node)
    graph.add_node("rank_sources", nodes.rank_sources_node)
    graph.add_node("fetch_sources", nodes.fetch_sources)
    graph.add_node("assess_evidence", nodes.assess_evidence)
    graph.add_node("refine_search", nodes.refine_search)
    graph.add_node("synthesize_answer", nodes.synthesize_answer)
    graph.add_node("validate_citations", nodes.validate_citations_node)
    graph.add_node("build_response", nodes.build_response)

    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {"valid": "analyze_query", "invalid": "build_response"},
    )
    graph.add_edge("analyze_query", "plan_search")
    graph.add_edge("plan_search", "search_providers")
    graph.add_edge("search_providers", "normalize_results")
    graph.add_edge("normalize_results", "deduplicate_results")
    graph.add_edge("deduplicate_results", "rank_sources")
    graph.add_edge("rank_sources", "fetch_sources")
    graph.add_edge("fetch_sources", "assess_evidence")
    graph.add_conditional_edges(
        "assess_evidence",
        route_after_evidence,
        {"refine": "refine_search", "synthesize": "synthesize_answer"},
    )
    graph.add_edge("refine_search", "search_providers")  # the one bounded cycle
    graph.add_edge("synthesize_answer", "validate_citations")
    graph.add_edge("validate_citations", "build_response")
    graph.add_edge("build_response", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()
