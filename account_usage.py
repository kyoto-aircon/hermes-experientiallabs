"""Experiential Labs account-usage helpers for the Hermes ``/usage`` snapshot.

The plugin's ``ProviderProfile.fetch_account_usage`` hook is dispatched by
Hermes only for providers without a built-in fetcher. This module turns the
Cost / Account API responses into ``AccountUsageWindow`` values:

- ``GET /api/v1/credits`` → ``{data: {total_credits, total_usage}}`` in USD;
  shown as the wallet balance window.
- ``GET /api/v1/key`` → the presented key's ``limit`` / ``limit_remaining``
  (``limit`` is a daily USD cap that resets at 00:00 UTC); shown as a second
  window when a cap is set.

The builders are pure (no I/O) so they can be unit-tested directly; the fetch
function is defensive and returns ``None`` on any failure so the caller fails
open exactly like the dispatch boundary expects.

Optional overrides (``cost_api_base``, ``usage_timeout_s``) may be set from
the Desktop/TUI plugin settings pages; they are read through Hermes' readonly
config loader lazily inside the fetch path and every failure — missing config,
unexpected shape, import error — falls back to the built-in defaults.

Only stdlib is imported at module load. ``httpx`` and Hermes internals are
imported lazily inside the fetch path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

_DEFAULT_COST_API_BASE = "https://api.experientiallabs.ai/api/v1"
_CREDITS_URL = f"{_DEFAULT_COST_API_BASE}/credits"
_KEY_URL = f"{_DEFAULT_COST_API_BASE}/key"
# Per-request timeout for the two sequential GETs; the worst case (2x) must
# stay under Hermes' 10s usage-hook deadline (2x4.5s leaves room for the
# resolver and thread hand-off), so user-provided values above
# ``_MAX_TIMEOUT_S`` are clamped.
_HTTP_TIMEOUT_S = 4.0
_MAX_TIMEOUT_S = 4.5
# Manifest name (plugin.yaml ``name``): optional settings live under
# ``plugins.entries.<id>.settings`` in config.yaml.
_PLUGIN_ID = "hermes-experientiallabs"

_MISSING = object()
_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


def _to_decimal(value: Any) -> Decimal | None:
    """Parse *value* into a finite ``Decimal`` or return ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def _pick(mapping: dict, *keys: str) -> Any:
    """First present, non-``None`` value among *keys*, else ``_MISSING``.

    Keeps the present-but-zero case distinct from "field absent" so a hard
    ``$0`` cap is not mistaken for "no cap configured".
    """
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return _MISSING


def _unwrap(payload: Any) -> dict:
    """Return the payload's ``data`` object, tolerating a bare top-level body."""
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload


def _usd(value: Decimal) -> str:
    return f"{value:.2f}"


def _window(*, label: str, used_percent: float | None = None,
            reset_at: datetime | None = None, detail: str | None = None) -> Any:
    # Imported lazily: keeps plugin import light and avoids pulling the Hermes
    # core account-usage layer during provider discovery.
    from agent.account_usage import AccountUsageWindow

    return AccountUsageWindow(
        label=label, used_percent=used_percent, reset_at=reset_at, detail=detail
    )


def next_utc_midnight(now: datetime | None = None) -> datetime:
    """The next 00:00 UTC boundary at or after *now* (the daily cap reset)."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    tomorrow = (current.astimezone(timezone.utc) + timedelta(days=1)).date()
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)


def build_wallet_window(payload: Any) -> Any | None:
    """Wallet balance window from a ``/credits`` response, or ``None``."""
    data = _unwrap(payload)
    total = _to_decimal(data.get("total_credits"))
    used = _to_decimal(data.get("total_usage"))
    if total is None or used is None:
        return None

    remaining = max(_ZERO, total - used)
    if total > 0:
        percent = float(max(_ZERO, min(_HUNDRED, used / total * _HUNDRED)))
    else:
        percent = 0.0
    detail = f"${_usd(remaining)} remaining (${_usd(used)} used)"
    return _window(label="Credits", used_percent=percent, detail=detail)


def build_daily_cap_window(payload: Any, *, now: datetime | None = None) -> Any | None:
    """Daily key-cap window from a ``/key`` response, or ``None`` when uncapped."""
    data = _unwrap(payload)
    raw_limit = _pick(data, "limit", "daily_limit", "cap")
    if raw_limit is _MISSING:
        return None
    limit = _to_decimal(raw_limit)
    if limit is None:
        return None

    reset_at = next_utc_midnight(now)
    if limit <= 0:
        # A non-positive limit is a hard $0 cap, never "unlimited". Checked
        # before requiring ``limit_remaining`` because a disabled/hard cap may
        # omit it entirely.
        return _window(label="Daily key cap", used_percent=100.0,
                       reset_at=reset_at, detail="hard $0 cap")

    raw_remaining = _pick(data, "limit_remaining", "remaining", "limitRemaining")
    if raw_remaining is _MISSING:
        return None
    remaining = _to_decimal(raw_remaining)
    if remaining is None:
        return None

    used = max(_ZERO, limit - remaining)
    percent = float(max(_ZERO, min(_HUNDRED, used / limit * _HUNDRED)))
    return _window(label="Daily key cap", used_percent=percent, reset_at=reset_at,
                   detail=f"${_usd(used)} of ${_usd(limit)} used")


def _cost_urls(cost_api_base: Any) -> tuple[str, str]:
    """``(credits_url, key_url)`` for a Cost API base, defaulting fail-open."""
    base = cost_api_base.strip().rstrip("/") if isinstance(cost_api_base, str) else ""
    if not base:
        return _CREDITS_URL, _KEY_URL
    return f"{base}/credits", f"{base}/key"


def _usage_timeout(value: Any) -> float:
    """Per-request timeout from a raw settings value, clamped and fail-open."""
    if value is None or isinstance(value, bool):
        return _HTTP_TIMEOUT_S
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return _HTTP_TIMEOUT_S
    if not timeout > 0 or not isfinite(timeout):
        return _HTTP_TIMEOUT_S
    # Two sequential GETs must fit Hermes' 10s usage-hook deadline.
    return min(timeout, _MAX_TIMEOUT_S)


def resolve_usage_settings() -> tuple[str, str, float]:
    """``(credits_url, key_url, timeout)`` for the fetch, fail-open.

    Reads ``plugins.entries.hermes-experientiallabs.settings`` — the values
    saved by the Desktop/TUI plugin settings pages — through Hermes' readonly
    config loader. The import is lazy (module load stays stdlib-only) and any
    failure falls back to the built-in defaults: the usage hook must fail
    open, never raise.
    """
    settings: Any = None
    try:
        from hermes_cli.config import load_config_readonly

        settings = (
            load_config_readonly()
            .get("plugins", {})
            .get("entries", {})
            .get(_PLUGIN_ID, {})
            .get("settings")
        )
    except Exception:  # noqa: BLE001 - optional overrides must fail open
        settings = None
    if not isinstance(settings, dict):
        return _CREDITS_URL, _KEY_URL, _HTTP_TIMEOUT_S
    credits_url, key_url = _cost_urls(settings.get("cost_api_base"))
    return credits_url, key_url, _usage_timeout(settings.get("usage_timeout_s"))


def _get_json(client: Any, url: str, headers: dict[str, str]) -> Any | None:
    """GET *url* returning the decoded JSON, or ``None`` on any failure."""
    try:
        response = client.get(url, headers=headers)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:  # noqa: BLE001 - a failed sub-request must not lose the other
        return None


def fetch_experientiallabs_account_usage(
    provider_name: str, *, base_url: str | None = None, api_key: str | None = None
) -> Any | None:
    """Build the ``/usage`` snapshot for Experiential Labs, or ``None``."""
    runtime = None
    try:
        import httpx

        from agent.account_usage import AccountUsageSnapshot
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested=provider_name, explicit_base_url=base_url, explicit_api_key=api_key
        )
    except Exception:  # noqa: BLE001 - imports and credentials must fail open
        return None

    token = str((runtime or {}).get("api_key", "") or "").strip()
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    credits_url, key_url, timeout_s = resolve_usage_settings()
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            credits = _get_json(client, credits_url, headers)
            key = _get_json(client, key_url, headers)
    except Exception:  # noqa: BLE001 - fail open
        return None

    windows = []
    for payload, builder in ((credits, build_wallet_window), (key, build_daily_cap_window)):
        if payload is None:
            continue
        window = builder(payload)
        if window is not None:
            windows.append(window)
    if not windows:
        return None

    return AccountUsageSnapshot(
        provider=provider_name,
        source="explabs_cost_api",
        fetched_at=datetime.now(timezone.utc),
        windows=tuple(windows),
    )
