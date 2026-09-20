import pytest

from app.retrieval.canonicalize import canonicalize_url, domain_of, strip_tracking_params


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://Example.COM/Docs#section", "https://example.com/Docs"),
        ("HTTPS://WWW.Example.com/a/b/", "https://example.com/a/b"),
        ("https://example.com:443/x", "https://example.com/x"),
        ("http://example.com:80/x", "http://example.com/x"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com/", "https://example.com/"),
        ("example.com/path", "https://example.com/path"),
    ],
)
def test_canonicalization_rules(raw, expected):
    assert canonicalize_url(raw) == expected


def test_fragment_is_removed():
    assert "#" not in canonicalize_url("https://example.com/a#anchor")


def test_path_case_is_preserved():
    assert canonicalize_url("https://Example.com/CaseSensitive") == (
        "https://example.com/CaseSensitive"
    )


def test_empty_and_invalid_inputs_do_not_raise():
    assert canonicalize_url("") == ""
    assert canonicalize_url(None) == ""


def test_domain_of_strips_www():
    assert domain_of("https://www.Arxiv.org/abs/1") == "arxiv.org"


@pytest.mark.parametrize(
    "param", ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid"]
)
def test_tracking_parameters_are_removed(param):
    url = f"https://example.com/page?{param}=abc&id=7"
    canonical = canonicalize_url(url)
    assert param not in canonical
    assert "id=7" in canonical


def test_meaningful_parameters_are_preserved():
    canonical = canonicalize_url("https://example.com/search?q=rag&page=2&utm_source=x")
    assert "q=rag" in canonical and "page=2" in canonical and "utm_source" not in canonical


def test_parameter_order_is_deterministic():
    assert canonicalize_url("https://e.com/a?b=2&a=1") == canonicalize_url(
        "https://e.com/a?a=1&b=2"
    )


def test_strip_tracking_params_on_empty_query():
    assert strip_tracking_params("") == ""
