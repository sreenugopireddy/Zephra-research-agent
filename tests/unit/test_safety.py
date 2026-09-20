import pytest

from app.validation.safety import is_safe_url, reject_reason


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "data:text/html,<script>",
        "javascript:alert(1)",
        "ftp://example.com/x",
        "http://localhost/admin",
        "http://127.0.0.1:8000/",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.5/internal",
        "",
    ],
)
def test_unsafe_urls_are_rejected(url):
    assert is_safe_url(url) is False
    assert reject_reason(url)


@pytest.mark.parametrize("url", ["https://example.com/a", "http://example.org/b?q=1"])
def test_public_http_urls_are_allowed(url):
    assert is_safe_url(url) is True
    assert reject_reason(url) == ""


def test_file_scheme_reason_is_explicit():
    assert "unsafe scheme" in reject_reason("file:///etc/passwd")
