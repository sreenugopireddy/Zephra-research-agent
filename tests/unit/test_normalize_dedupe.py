from app.retrieval.deduplicate import deduplicate_results
from app.retrieval.normalize import clean_text, normalize_results
from app.schemas.search import SearchResult
from tests.fixtures.fakes import google_results, tavily_results


def test_normalization_preserves_provenance():
    normalized = normalize_results(tavily_results())
    first = normalized[0]
    assert first.provider == "tavily"
    assert first.provider_rank == 0
    assert first.provider_score == 0.93
    assert first.matched_query == ""
    assert first.canonical_url == "https://arxiv.org/abs/2401.00001"
    assert first.domain == "arxiv.org"


def test_normalization_strips_markup_and_entities():
    assert clean_text("<b>Hello</b>&amp;bye") == "Hello &bye"


def test_results_without_a_url_are_dropped():
    rows = [SearchResult(url="   ", provider="tavily"), *tavily_results()]
    assert len(normalize_results(rows)) == 2


def test_same_url_from_two_providers_is_one_source():
    normalized = normalize_results(tavily_results() + google_results())
    sources = deduplicate_results(normalized)
    urls = [s.canonical_url for s in sources]
    assert len(urls) == len(set(urls))
    merged = next(s for s in sources if "arxiv.org" in s.canonical_url)
    assert sorted(merged.found_by) == ["google", "tavily"]
    assert merged.duplicate_count == 2


def test_merge_preserves_all_useful_metadata():
    sources = deduplicate_results(normalize_results(tavily_results() + google_results()))
    merged = next(s for s in sources if "arxiv.org" in s.canonical_url)
    assert len(merged.snippets) == 2
    assert merged.provider_ranks == {"tavily": 0, "google": 0}
    assert merged.provider_scores["tavily"] == 0.93
    assert merged.published_date == "2024-06-01"


def test_source_ids_are_sequential_and_stable():
    sources = deduplicate_results(normalize_results(tavily_results() + google_results()))
    assert [s.source_id for s in sources] == [f"S{i}" for i in range(1, len(sources) + 1)]


def test_redirect_target_merges_sources():
    rows = [
        SearchResult(url="https://short.link/abc", provider="tavily", title="A shortened link",
                     redirect_target="https://example.com/real-article"),
        SearchResult(url="https://example.com/real-article", provider="google",
                     title="The real article"),
    ]
    sources = deduplicate_results(normalize_results(rows))
    assert len(sources) == 1
    assert sorted(sources[0].found_by) == ["google", "tavily"]


def test_matching_title_and_domain_merges():
    rows = [
        SearchResult(url="https://site.com/a?id=1", provider="tavily",
                     title="A Sufficiently Long Article Title"),
        SearchResult(url="https://site.com/b?id=2", provider="google",
                     title="A Sufficiently Long Article Title"),
    ]
    assert len(deduplicate_results(normalize_results(rows))) == 1


def test_short_titles_do_not_falsely_merge():
    rows = [
        SearchResult(url="https://site.com/a", provider="tavily", title="Docs"),
        SearchResult(url="https://site.com/b", provider="google", title="Docs"),
    ]
    assert len(deduplicate_results(normalize_results(rows))) == 2
