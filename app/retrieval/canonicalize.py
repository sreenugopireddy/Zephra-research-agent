"""Deterministic URL canonicalization.

Rules (applied in order):
  1. Trim whitespace, add an https:// scheme when missing.
  2. Lowercase the scheme and the hostname (the path stays case-sensitive).
  3. Drop the fragment (#section).
  4. Drop default ports (:80 for http, :443 for https).
  5. Strip a leading 'www.' so www.x.com and x.com collapse together.
  6. Remove tracking parameters; keep every other query parameter, sorted
     so that parameter order never produces two "different" URLs.
  7. Normalize the trailing slash: removed everywhere except the site root.
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_name",
        "gclid",
        "gclsrc",
        "dclid",
        "fbclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "ref_src",
        "_ga",
        "yclid",
    }
)

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def strip_tracking_params(query: str) -> str:
    """Remove known tracking parameters, preserving meaningful ones (sorted)."""
    if not query:
        return ""
    kept = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    kept.sort()
    return urlencode(kept)


def canonicalize_url(url: str) -> str:
    """Return a stable canonical form of `url`. Never raises."""
    if not url or not isinstance(url, str):
        return ""
    raw = url.strip()
    if not raw:
        return ""
    if "//" not in raw.split("?", 1)[0][:8]:
        raw = "https://" + raw.lstrip("/")

    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw

    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    netloc = host
    if parts.port and str(parts.port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    return urlunsplit((scheme, netloc, path, strip_tracking_params(parts.query), ""))


def domain_of(url: str) -> str:
    """Registrable-ish hostname of a URL, lowercased and without 'www.'."""
    try:
        host = (urlsplit(url if "//" in url else "https://" + url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host
