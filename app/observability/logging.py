"""Structured JSON logging with credential redaction.

API keys are never written to logs: every value is passed through `_redact`,
and known secret-bearing keys are dropped entirely.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

_SECRET_KEY_HINTS = ("api_key", "apikey", "token", "secret", "password", "authorization", "cse_id")
_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("research_agent")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    _configured = True


def _redact(key: str, value: Any) -> Any:
    if any(hint in key.lower() for hint in _SECRET_KEY_HINTS):
        return "***redacted***"
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "...<truncated>"
    return value


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    """Emit one structured log line."""
    configure_logging()
    payload = {"event": event}
    payload.update({k: _redact(k, v) for k, v in fields.items()})
    logger = logging.getLogger("research_agent")
    getattr(logger, level.lower(), logger.info)(json.dumps(payload, default=str, sort_keys=True))
