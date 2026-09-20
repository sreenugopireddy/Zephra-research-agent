"""URL safety checks applied before any network fetch."""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})
BLOCKED_SCHEMES = frozenset({"file", "ftp", "data", "javascript", "gopher", "mailto", "about"})
_BLOCKED_HOSTNAMES = frozenset({"localhost", "metadata.google.internal", "169.254.169.254"})


def is_safe_url(url: str) -> bool:
    """Return True only for public http(s) URLs.

    Rejects unsafe schemes (file://, data:, javascript:), loopback, link-local
    and private addresses, so that source fetching cannot read local files or
    reach internal services.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return False
    host = (parts.hostname or "").lower()
    if not host or host in _BLOCKED_HOSTNAMES:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True  # a regular hostname
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    )


def reject_reason(url: str) -> str:
    """Human-readable reason a URL was rejected (empty string when safe)."""
    if is_safe_url(url):
        return ""
    scheme = urlsplit(url).scheme.lower() if url else ""
    if scheme in BLOCKED_SCHEMES:
        return f"unsafe scheme '{scheme}'"
    if scheme not in ALLOWED_SCHEMES:
        return "url must use http or https"
    return "host is not publicly routable"
