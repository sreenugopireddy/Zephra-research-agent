from app.retrieval.rank import (
    WEIGHTS,
    rank_sources,
    score_authority,
    score_freshness,
    score_provider_agreement,
)
from app.schemas.search import DeduplicatedSource


def make_source(source_id="S1", domain="example.com", **kwargs) -> DeduplicatedSource:
    defaults = dict(
        source_id=source_id,
        title="RAG and fine-tuning trade-offs for enterprise customer support",
        url=f"https://{domain}/x",
        canonical_url=f"https://{domain}/x",
        domain=domain,
        snippets=["rag fine-tuning enterprise customer support trade-offs discussion"],
        found_by=["tavily"],
        matched_queries=["rag vs fine-tuning"],
        provider_ranks={"tavily": 0},
    )
    defaults.update(kwargs)
    return DeduplicatedSource(**defaults)


QUESTION = "trade-offs between RAG and fine-tuning for enterprise customer support"


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_scores_stay_within_bounds():
    ranked = rank_sources([make_source(), make_source("S2", "arxiv.org")], QUESTION)
    for source in ranked:
        assert 0.0 <= source.score <= 1.0


def test_authority_tiers_are_ordered():
    assert score_authority("nih.gov") > score_authority("docs.python.org")
    assert score_authority("docs.python.org") > score_authority("github.com")
    assert score_authority("github.com") > score_authority("pinterest.com")


def test_authoritative_domain_outranks_low_quality_domain():
    ranked = rank_sources(
        [make_source("S1", "pinterest.com"), make_source("S2", "nih.gov")], QUESTION
    )
    assert ranked[0].source_id == "S2"


def test_multi_provider_agreement_raises_the_score():
    single = rank_sources([make_source("S1")], QUESTION)[0]
    both = rank_sources([make_source("S1", found_by=["tavily", "google"])], QUESTION)[0]
    assert both.score > single.score
    assert score_provider_agreement(make_source(found_by=["tavily", "google"])) == 1.0


def test_freshness_is_neutral_when_not_required():
    assert score_freshness(None, freshness_required=False) == 0.5
    assert score_freshness("2020-01-01", freshness_required=True) < score_freshness(
        "2025-09-01", freshness_required=True
    )


def test_relevance_rewards_topic_overlap():
    on_topic = make_source("S1")
    off_topic = make_source(
        "S2", title="Unrelated gardening tips", snippets=["how to grow tomatoes"]
    )
    ranked = rank_sources([off_topic, on_topic], QUESTION)
    assert ranked[0].source_id == "S1"


def test_breakdown_is_exposed_for_every_component():
    ranked = rank_sources([make_source()], QUESTION)[0]
    assert set(ranked.score_breakdown) == set(WEIGHTS)


def test_ranking_is_deterministic():
    sources = [make_source("S1"), make_source("S2", "arxiv.org"), make_source("S3", "nih.gov")]
    first = [s.source_id for s in rank_sources(sources, QUESTION)]
    second = [s.source_id for s in rank_sources(sources, QUESTION)]
    assert first == second
