import httpx

from app.retrieval.fetch import extract_text, fetch_documents, fetch_source
from app.schemas.search import ExtractionStatus, RankedSource

LONG_HTML = (
    "<html><head><title>Doc title</title></head><body>"
    "<script>evil()</script><style>.a{}</style>"
    "<p>" + ("Useful evidence sentence. " * 60) + "</p></body></html>"
)


def make_source(url="https://example.org/page", snippet="fallback snippet text") -> RankedSource:
    return RankedSource(
        source_id="S1",
        title="Page",
        url=url,
        canonical_url=url,
        domain="example.org",
        snippets=[snippet],
        found_by=["tavily"],
    )


def test_extract_text_drops_scripts_and_styles():
    text, title = extract_text(LONG_HTML)
    assert "evil()" not in text and ".a{}" not in text
    assert title == "Doc title"


def test_extract_text_survives_malformed_html():
    text, _ = extract_text("<p>unclosed <b>tags")
    assert "unclosed" in text


async def test_successful_fetch_marks_success_and_truncates():
    def handler(request):
        return httpx.Response(200, text=LONG_HTML, headers={"content-type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        doc = await fetch_source(client, make_source(), max_chars=500, timeout=5)

    assert doc.extraction_status == ExtractionStatus.success
    assert doc.content_chars <= 500
    assert doc.retrieved_at and doc.source_id == "S1"


async def test_http_error_falls_back_to_snippet_as_partial():
    def handler(request):
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        doc = await fetch_source(client, make_source(), max_chars=500, timeout=5)

    assert doc.extraction_status == ExtractionStatus.partial
    assert doc.content == "fallback snippet text"
    assert doc.error


async def test_page_without_snippet_and_without_content_is_failed():
    def handler(request):
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        doc = await fetch_source(client, make_source(snippet=""), max_chars=500, timeout=5)

    assert doc.extraction_status == ExtractionStatus.failed
    assert doc.usable is False


async def test_unsafe_url_is_never_requested():
    called = {"n": 0}

    def handler(request):
        called["n"] += 1
        return httpx.Response(200, text=LONG_HTML)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        doc = await fetch_source(client, make_source("file:///etc/passwd"), 500, 5)

    assert called["n"] == 0
    assert "unsafe scheme" in (doc.error or "")


async def test_thin_page_is_marked_partial():
    def handler(request):
        return httpx.Response(200, text="<p>tiny</p>", headers={"content-type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        doc = await fetch_source(client, make_source(), max_chars=500, timeout=5)

    assert doc.extraction_status == ExtractionStatus.partial


async def test_failures_are_isolated_per_page(monkeypatch):
    sources = [make_source("https://a.org/1"), make_source("https://b.org/2")]
    sources[1].source_id = "S2"

    async def fake_fetch(client, source, max_chars, timeout):
        if source.source_id == "S1":
            raise RuntimeError("boom")
        from app.retrieval.fetch import _snippet_document

        return _snippet_document(source, "ok")

    monkeypatch.setattr("app.retrieval.fetch.fetch_source", fake_fetch)
    docs = await fetch_documents(sources)
    assert [d.source_id for d in docs] == ["S1", "S2"]
    assert docs[0].extraction_status == ExtractionStatus.partial


async def test_empty_source_list_returns_empty():
    assert await fetch_documents([]) == []
