"""Pure 429 reset-hint extraction for the Experiential Labs classify hook (W3).

The profile's ``classify_api_error`` runs BEFORE Hermes' built-in 429 handler
(``agent/error_classifier.py::_profile_verdict``), so a plugin rate-limit
verdict must carry its own reset window or the retry-exhausted terminal copy
cannot name it: ``agent/turn_recovery.py::max_retries_exhausted_result`` reads
``classified.error_context["reset_at"]`` as ABSOLUTE EPOCH SECONDS
(``reset_at - time.time()``), the same shape ``agent/nous_rate_guard.py``
(``isinstance(ctx_reset, (int, float)) and ctx_reset > now``) and
``agent/fallback_cooldown.py`` (via ``_parse_absolute_timestamp``) expect.
This module mirrors the built-in ``_status_429`` context
(``{"reset_at": time.time() + reset}``).

Retry BACKOFF needs no help from the plugin: ``compute_error_backoff`` honours
``Retry-After`` straight off the SDK error
(``agent/retry_utils.py::provider_retry_after_seconds``), and the turn's own
``extract_api_error_context`` already parses the raw headers for fallback
cooldowns. The gap this fills is the classifier verdict's ``error_context``.

Gateway contract (platform.experientiallabs.ai/llms.txt): the 429 codes
``rate_limit_exceeded``, ``admission_capacity_exceeded`` and the hourly/daily
``free_limit_reached`` (type ``rate_limit_error``) carry ``Retry-After``
seconds; the ``free_limit_reached`` message also names the window's reset.
OpenAI-style ``x-ratelimit-reset-*`` headers are accepted defensively as a
secondary source but are not documented gateway behaviour.

Rules honoured here: pure (no I/O), stdlib only at module load, never raises —
a malformed or missing hint must cost the hint, never the verdict.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

# Malformed-value guard: no honest rate-limit window is a year out (the
# gateway's windows are per-minute/hourly/daily). Anything beyond this is a
# broken header, not a reset hint.
_MAX_RESET_WINDOW_S = 366 * 24 * 3600

# Header names probed when a headers object exposes .get() but not .items().
_CANONICAL_HEADER_SPELLINGS = {
    "retry-after": "Retry-After",
    "x-ratelimit-reset": "X-RateLimit-Reset",
    "x-ratelimit-reset-requests": "X-RateLimit-Reset-Requests",
    "x-ratelimit-reset-requests-1h": "X-RateLimit-Reset-Requests-1h",
    "x-ratelimit-reset-tokens": "X-RateLimit-Reset-Tokens",
    "x-ratelimit-reset-tokens-1h": "X-RateLimit-Reset-Tokens-1h",
}

# OpenAI-style relative windows: "6m0s", "1.5s", "20ms", "1h2m3s" (also a bare
# number of seconds). Mirrors agent/agent_runtime_helpers.py::_duration_string_seconds.
_DURATION_COMPONENT_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}

# Free-text reset grammars providers put in error messages, mirroring
# agent/retry_utils.py (explicit "retry after N s" wins over "resets in ...").
_RETRY_AFTER_TEXT_RE = re.compile(
    r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)", re.IGNORECASE
)
_RESETS_IN_TEXT_RE = re.compile(
    r"resets?\s+in\s+"
    r"(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b\s*)?"
    r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b\s*)?"
    r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b)?",
    re.IGNORECASE,
)


def _retry_after_seconds(value: Any) -> float | None:
    """Seconds from a ``Retry-After`` value: number, numeric string or HTTP-date.

    Mirrors ``agent/retry_utils.py::parse_retry_after_seconds`` for a single
    value (headers mappings are normalised before they get here). Unparseable,
    empty, boolean and non-finite values yield None; callers reject <= 0.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        return num if math.isfinite(num) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        num = float(text)
    except (TypeError, ValueError):
        num = None
    if num is not None:
        return num if math.isfinite(num) else None
    try:
        when = parsedate_to_datetime(text)
    except Exception:  # noqa: BLE001 - stdlib raises assorted types on junk dates
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    try:
        return (when - datetime.now(timezone.utc)).total_seconds()
    except Exception:  # noqa: BLE001 - overflow on extreme dates
        return None


def _absolute_epoch(value: Any) -> float | None:
    """Epoch seconds from epoch s / epoch ms / ISO-8601.

    Mirrors ``agent/credential_pool_timestamps.py::_parse_absolute_timestamp``
    (the parser downstream consumers apply to ``reset_at``), with naive ISO
    timestamps read as UTC so the hint is machine-independent.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            num = float(text)
        except (TypeError, ValueError):
            iso = text[:-1] + "+00:00" if text.endswith("Z") else text
            try:
                when = datetime.fromisoformat(iso)
            except (TypeError, ValueError):
                return None
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            try:
                return when.timestamp()
            except (OverflowError, OSError, ValueError):
                return None
    if not math.isfinite(num) or num <= 0:
        return None
    return num / 1000.0 if num > 1_000_000_000_000 else num


def _duration_seconds(value: Any) -> float | None:
    """Relative seconds from a number, numeric string or OpenAI duration string."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        return num if math.isfinite(num) else None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        num = float(text)
    except (TypeError, ValueError):
        parts = _DURATION_COMPONENT_RE.findall(text)
        if not parts or "".join(n + u for n, u in parts) != text:
            return None
        return sum(float(n) * _DURATION_UNIT_SECONDS[u] for n, u in parts)
    return num if math.isfinite(num) else None


def _lower_headers(headers: Any) -> dict[str, Any]:
    """Lowercase a headers mapping defensively (dict, httpx.Headers, oddballs).

    Never raises: a headers object whose ``items()``/``get()`` misbehaves
    yields an empty mapping, costing only the header-sourced hints.
    """
    if headers is None:
        return {}
    try:
        return {str(k).lower(): v for k, v in headers.items()}
    except Exception:  # noqa: BLE001
        pass
    out: dict[str, Any] = {}
    getter = getattr(headers, "get", None)
    if callable(getter):
        for name, canonical in _CANONICAL_HEADER_SPELLINGS.items():
            for spelling in (name, canonical):
                try:
                    value = getter(spelling)
                except Exception:  # noqa: BLE001
                    continue
                if value is not None:
                    out[name] = value
                    break
    return out


def _reset_seconds_from_message(text: str) -> float | None:
    """Seconds from the free-text reset grammars (``retry after N s`` first)."""
    match = _RETRY_AFTER_TEXT_RE.search(text)
    if match:
        return float(match.group(1))
    match = _RESETS_IN_TEXT_RE.search(text)
    if match and any(match.groups()):
        return (
            float(match.group(1) or 0) * 3600
            + float(match.group(2) or 0) * 60
            + float(match.group(3) or 0)
        )
    return None


def _bounded(epoch: float | None, now: float) -> float | None:
    """Keep only finite, future, sane reset epochs (expired/zero hints are noise)."""
    if epoch is None or not math.isfinite(epoch):
        return None
    if epoch <= now or epoch > now + _MAX_RESET_WINDOW_S:
        return None
    return epoch


def reset_epoch_from_error(
    error: BaseException | None,
    *,
    body: object = None,
    message: str | None = None,
) -> float | None:
    """Absolute epoch seconds when the gateway says the 429 window reopens.

    Sources, in priority order (mirroring core's ``extract_api_error_context``
    and ``_rate_limit_reset_seconds``): the body's ``resets_at``/``reset_at``
    (epoch s/ms or ISO-8601), ``resets_in_seconds``, ``retry_after``; then the
    raw error's ``response.headers`` ``Retry-After`` (seconds or HTTP-date),
    the OpenAI-style ``x-ratelimit-reset-*`` relative windows and
    ``x-ratelimit-reset`` epoch; then the message grammars. None when nothing
    parseable and future remains — never raises.
    """
    try:
        now = time.time()
        raw_body = body if isinstance(body, dict) else getattr(error, "body", None)
        payloads: list[dict] = []
        if isinstance(raw_body, dict):
            inner = raw_body.get("error")
            if isinstance(inner, dict):
                payloads.append(inner)
            payloads.append(raw_body)

        for payload in payloads:
            for field in ("resets_at", "reset_at"):
                epoch = _bounded(_absolute_epoch(payload.get(field)), now)
                if epoch is not None:
                    return epoch
        for payload in payloads:
            seconds = _duration_seconds(payload.get("resets_in_seconds"))
            if seconds is not None and seconds > 0:
                return _bounded(now + seconds, now)
        for payload in payloads:
            seconds = _retry_after_seconds(payload.get("retry_after"))
            if seconds is not None and seconds > 0:
                return _bounded(now + seconds, now)

        headers = _lower_headers(
            getattr(getattr(error, "response", None), "headers", None)
        )
        seconds = _retry_after_seconds(headers.get("retry-after"))
        if seconds is not None and seconds > 0:
            return _bounded(now + seconds, now)
        for name in (
            "x-ratelimit-reset-requests-1h",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens-1h",
            "x-ratelimit-reset-tokens",
        ):
            seconds = _duration_seconds(headers.get(name))
            if seconds is not None and seconds > 0:
                return _bounded(now + seconds, now)
        epoch = _bounded(_absolute_epoch(headers.get("x-ratelimit-reset")), now)
        if epoch is not None:
            return epoch

        texts = [message]
        try:
            texts.append(str(error) if error is not None else "")
        except Exception:  # noqa: BLE001 - exotic __str__
            texts.append("")
        texts.extend(str(p.get("message") or "") for p in payloads)
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            seconds = _reset_seconds_from_message(text)
            if seconds is not None and seconds > 0:
                return _bounded(now + seconds, now)
        return None
    except Exception:  # noqa: BLE001 - a hint must never cost the verdict
        return None


def rate_limit_error_context(
    error: BaseException | None,
    *,
    body: object = None,
    message: str | None = None,
) -> dict[str, float]:
    """``{"reset_at": <epoch s>}`` for a rate-limit verdict, or ``{}`` when none.

    The shape Nous' guard and the retry-exhausted copy consume: an absolute,
    future epoch-seconds float under ``reset_at``.
    """
    epoch = reset_epoch_from_error(error, body=body, message=message)
    return {"reset_at": epoch} if epoch is not None else {}
