"""Live catalog metadata, reasoning capabilities, context limits, and pricing.

Mirrors the built-in Nous Portal provider's catalog layer (``models_reasoning_caps``
and ``models_pricing``) inside the plugin:

- ``GET /v1/models`` on Experiential Labs already returns per-model
  ``context_window_tokens``, ``maximum_output_tokens``, ``supports_tools``,
  ``supports_completions``, ``supports_reasoning``, ``supported_reasoning_efforts``,
  ``reasoning_effort`` (default), ``chat_max_tokens_field``, and ``pricing``
  (nano-USD per million tokens, including cache read/write and long-context tiers).
- ``fetch_v1_catalog`` fetches ``/v1/models`` once during model listing, filters
  out non-chat/non-tool entries (embeddings, TTS, OCR, RP-only routes), and
  seeds an in-memory + disk mirror cache at zero extra network cost.
- Hot-path hooks (``get_model_context_length``, ``supported_reasoning_efforts``,
  ``build_api_kwargs_extras``, ``max_tokens_param``, ``get_usage_cost``) read
  from memory / disk cache only and NEVER block on HTTP; when cold, they kick a
  background warmer (skipped under pytest) and fail open.
- ``live_context_length`` can fall back to the keyless public detail endpoint
  ``GET /api/models/<slug>`` which reports ``max_input_tokens`` (accounting for
  output reservation on makers whose total context includes max output).

Only stdlib is imported at module load; ``httpx`` and Hermes internals are
imported lazily inside I/O or pricing paths.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

_DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
_PUBLIC_MODEL_DETAIL_BASE = "https://api.experientiallabs.ai/api/models"
_DISK_CACHE_FILENAME = "experientiallabs_catalog.json"
_DISK_TTL_S = 24 * 3600
_FAIL_COOLDOWN_S = 60.0
_LIVE_DETAIL_TIMEOUT_S = 5.0

# Canonical low -> high reasoning effort ladder (mirrors agent.reasoning_effort).
EFFORT_LADDER: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)
_WIRE_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
_POSITIVE_WIRE_EFFORTS: tuple[str, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
_KNOWN_WIRE_EFFORT_SET = frozenset(_WIRE_EFFORTS)
_KNOWN_MAX_TOKENS_FIELDS = frozenset({"max_tokens", "max_completion_tokens"})

_ZERO = Decimal("0")
_NANO_PER_MILLION_DIVISOR = Decimal("1000000000000000")  # 1e9 nano-USD * 1e6 tokens
_SUBCENT_THRESHOLD = Decimal("0.01")
_FAST_MULTIPLIER = Decimal("2")

_FREE_ALLOWANCE_NOTE = (
    ":free routes draw on the gateway's promotional free allowance "
    "(hourly/daily caps), not wallet credits."
)
_BYOK_ESTIMATE_NOTE = (
    "BYOK pass-through: estimated upstream inference cost "
    "(not billed to Experiential Labs credits)."
)
_BYOK_ZERO_NOTE = (
    "BYOK pass-through: billed by your upstream provider, "
    "$0 Experiential Labs credits."
)
_MODELS_API_ESTIMATE_NOTE = (
    "Estimated from Experiential Labs /v1/models rate card."
)

# Promotional / routing suffixes the gateway accepts on a base model slug.
_ROUTING_SUFFIXES = (":free", ":fast", ":online")


def _to_decimal(value: Any) -> Decimal | None:
    """Parse *value* into a finite ``Decimal`` or return ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def _to_positive_int(value: Any) -> int | None:
    """Parse *value* into a positive ``int`` (excluding ``bool``) or ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _to_nonneg_nano_str(value: Any) -> str | None:
    """Normalize a non-negative nano-USD rate to a compact decimal string."""
    dec = _to_decimal(value)
    if dec is None or dec < _ZERO:
        return None
    return str(dec)


def split_model_suffix(model: str | None) -> tuple[str, str]:
    """Split a routing suffix (``:free``, ``:fast``, ``:online``) off *model*.

    Returns ``(base_slug, suffix)`` where ``suffix`` is lowercased (or ``""``).
    """
    if not isinstance(model, str):
        return "", ""
    cleaned = model.strip()
    lower = cleaned.lower()
    for suffix in _ROUTING_SUFFIXES:
        if lower.endswith(suffix):
            return cleaned[: -len(suffix)].strip(), suffix
    return cleaned, ""


def base_model_slug(model: str | None) -> str:
    """Base catalog slug for *model* with routing suffixes stripped."""
    base, _ = split_model_suffix(model)
    return base


def _lookup_candidates(model: str | None) -> tuple[str, ...]:
    """Candidate catalog keys in priority order (exact, suffix-stripped, vendor-stripped)."""
    if not isinstance(model, str):
        return ()
    raw = model.strip()
    if not raw:
        return ()
    out: list[str] = []

    def _add(val: str) -> None:
        if val and val not in out:
            out.append(val)

    _add(raw)
    base, _ = split_model_suffix(raw)
    _add(base)
    if "/" in base:
        _add(base.rsplit("/", 1)[-1].strip())
    return tuple(out)


def is_chat_tool_model(item: Any) -> bool:
    """True when a ``/v1/models`` row is usable for Hermes chat/tool calling.

    Filters out embeddings, TTS, OCR, and non-tool chat routes explicitly
    marked by the gateway. Missing capability booleans fail open (True).
    """
    if not isinstance(item, dict):
        return False
    mid = item.get("id")
    if not isinstance(mid, str) or not mid.strip():
        return False
    if item.get("supports_completions") is False:
        return False
    if item.get("supports_tools") is False:
        return False
    if item.get("supports_embeddings") is True and item.get("supports_completions") is not True:
        return False
    return True


def _parse_pricing_tier(raw: Any) -> dict[str, Any] | None:
    """Extract normalized nano-USD per-million rate strings from a pricing dict."""
    if not isinstance(raw, dict):
        return None
    mapping = {
        "input": ("input_nano_usd_per_million_tokens", "input_nano_usd_per_million"),
        "output": ("output_nano_usd_per_million_tokens", "output_nano_usd_per_million"),
        "cached_input": (
            "cached_input_nano_usd_per_million_tokens",
            "cached_input_nano_usd_per_million",
        ),
        "cache_write": (
            "cache_creation_input_nano_usd_per_million_tokens",
            "cache_write_nano_usd_per_million_tokens",
            "cache_write_input_nano_usd_per_million",
        ),
        "cache_write_1h": (
            "cache_creation_1h_input_nano_usd_per_million_tokens",
            "cache_write_1h_input_nano_usd_per_million",
        ),
        "reasoning": (
            "reasoning_nano_usd_per_million_tokens",
            "reasoning_nano_usd_per_million",
        ),
    }
    parsed: dict[str, Any] = {}
    for dest, keys in mapping.items():
        for key in keys:
            if key in raw and raw[key] is not None:
                val = _to_nonneg_nano_str(raw[key])
                if val is not None:
                    parsed[dest] = val
                    break
    return parsed or None


def parse_v1_model_item(item: Any) -> dict[str, Any] | None:
    """Normalize one Experiential Labs ``/v1/models`` row into a compact dict.

    Pure and JSON-serializable so it mirrors cleanly to disk.
    """
    if not isinstance(item, dict):
        return None
    mid = str(item.get("id") or "").strip()
    if not mid:
        return None

    entry: dict[str, Any] = {"id": mid}

    ctx = _to_positive_int(item.get("max_input_tokens")) or _to_positive_int(
        item.get("context_window_tokens")
    )
    if ctx is not None:
        entry["context_window"] = ctx

    max_out = _to_positive_int(item.get("maximum_output_tokens"))
    if max_out is not None:
        entry["max_output_tokens"] = max_out

    for flag in ("supports_tools", "supports_completions", "supports_reasoning"):
        val = item.get(flag)
        if isinstance(val, bool):
            entry[flag] = val

    raw_efforts = item.get("supported_reasoning_efforts")
    if isinstance(raw_efforts, list):
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw_eff in raw_efforts:
            eff = str(raw_eff or "").strip().lower()
            if eff in _KNOWN_WIRE_EFFORT_SET and eff not in seen:
                seen.add(eff)
                cleaned.append(eff)
        entry["supported_reasoning_efforts"] = cleaned

    def_eff = str(item.get("reasoning_effort") or "").strip().lower()
    if def_eff in _KNOWN_WIRE_EFFORT_SET:
        entry["default_reasoning_effort"] = def_eff

    mt_field = str(item.get("chat_max_tokens_field") or "").strip()
    if mt_field in _KNOWN_MAX_TOKENS_FIELDS:
        entry["chat_max_tokens_field"] = mt_field

    pricing_raw = item.get("pricing")
    pricing = _parse_pricing_tier(pricing_raw)
    if pricing:
        lc_raw = pricing_raw.get("long_context") if isinstance(pricing_raw, dict) else None
        if isinstance(lc_raw, dict):
            threshold = _to_positive_int(
                lc_raw.get("input_threshold_tokens")
                or lc_raw.get("long_context_threshold_tokens")
            )
            lc_tier = _parse_pricing_tier(lc_raw)
            if threshold and lc_tier:
                lc_tier["threshold"] = threshold
                pricing["long_context"] = lc_tier
        entry["pricing"] = pricing

    return entry


def parse_v1_catalog(items: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Parse ``/v1/models`` ``data`` array into ``(chat_tool_ids, entries_by_id)``."""
    if not isinstance(items, list):
        return [], {}
    ids: list[str] = []
    seen_ids: set[str] = set()
    entries: dict[str, dict[str, Any]] = {}
    for item in items:
        parsed = parse_v1_model_item(item)
        if parsed is None:
            continue
        mid = parsed["id"]
        entries[mid] = parsed
        if is_chat_tool_model(item) and mid.lower() not in seen_ids:
            seen_ids.add(mid.lower())
            ids.append(mid)
    return ids, entries


def parse_public_model_detail_context(payload: Any) -> int | None:
    """Extract input context bound from ``GET /api/models/<slug>`` payload.

    Prefers ``max_input_tokens`` (accounts for output reservation per gateway
    docs), falling back to ``context_window``.
    """
    if not isinstance(payload, dict):
        return None
    raw_model = payload.get("model")
    model_obj = raw_model if isinstance(raw_model, dict) else payload
    return _to_positive_int(model_obj.get("max_input_tokens")) or _to_positive_int(
        model_obj.get("context_window")
    )


# ── Reasoning clamping & resolution (pure) ─────────────────────────────


def clamp_effort_to_supported(
    effort: str | None, supported: Sequence[str] | None
) -> str | None:
    """Clamp *effort* to nearest weaker supported level (monotonic, never escalates).

    Matches ``agent.reasoning_effort.clamp_effort``: an enabled request is never
    clamped down to ``"none"``; if no weaker positive level exists, picks the
    weakest positive level.
    """
    requested = str(effort or "").strip().lower()
    if not requested:
        return None
    if not supported:
        return requested
    norm = [
        lvl
        for lvl in (str(s).strip().lower() for s in supported)
        if lvl in EFFORT_LADDER
    ]
    if not norm:
        return requested
    if requested in norm:
        return requested
    if requested not in EFFORT_LADDER:
        return requested
    candidates = [lvl for lvl in norm if lvl != "none"]
    if not candidates:
        return None
    req_idx = EFFORT_LADDER.index(requested)
    below = [lvl for lvl in candidates if EFFORT_LADDER.index(lvl) < req_idx]
    if below:
        return max(below, key=EFFORT_LADDER.index)
    return min(candidates, key=EFFORT_LADDER.index)


def entry_supported_reasoning_efforts(
    entry: dict[str, Any] | None,
) -> tuple[str, ...] | None:
    """Tri-state reasoning effort vocabulary from a catalog entry.

    - ``None``: unknown (entry absent or reasoning fields absent)
    - ``()``: model accepts NO reasoning controls (``supports_reasoning is False``)
    - non-empty tuple: accepted wire levels
    """
    if not isinstance(entry, dict):
        return None
    if entry.get("supports_reasoning") is False:
        return ()
    efforts = entry.get("supported_reasoning_efforts")
    if isinstance(efforts, (list, tuple)):
        if efforts:
            return tuple(efforts)
        if entry.get("supports_reasoning") is not True:
            return ()
    return None


def resolve_reasoning_effort(
    model: str | None,
    reasoning_config: dict[str, Any] | None,
    *,
    entry: dict[str, Any] | None,
) -> str | None:
    """Compute the top-level ``reasoning_effort`` value to send, or ``None`` to omit.

    Parity with Nous Portal's mandatory-reasoning omission and OpenRouter's
    catalog clamping:
    - Unset ``reasoning_config`` -> ``None`` (gateway uses the model's default).
    - Disable (``enabled is False`` or ``effort == "none"``):
      send ``"none"`` ONLY when catalog confirms ``"none"`` is accepted;
      omit when mandatory, non-reasoning, or cache-cold (omit beats HTTP 400).
    - Explicit effort: omit on non-reasoning models; clamp to the model's
      positive supported efforts when known, else clamp to the gateway's max
      wire vocabulary (so ``ultra`` -> ``max``).
    """
    if not isinstance(reasoning_config, dict) or not reasoning_config:
        return None

    raw_effort = str(reasoning_config.get("effort") or "").strip().lower()
    disabled = reasoning_config.get("enabled") is False or raw_effort == "none"

    supported = entry_supported_reasoning_efforts(entry)

    if disabled:
        if supported is not None and "none" in supported:
            return "none"
        return None

    if not raw_effort:
        return None

    if supported is not None:
        if not supported:
            return None
        positive = [lvl for lvl in supported if lvl != "none"]
        if not positive:
            return None
        return clamp_effort_to_supported(raw_effort, positive)

    return clamp_effort_to_supported(raw_effort, _POSITIVE_WIRE_EFFORTS)


# ── Cost calculation (pure) ────────────────────────────────────────────


def _format_cost_label(amount: Decimal, *, actual: bool = False) -> str:
    """Format a USD Decimal amount matching Hermes' ``format_cost_label``."""
    prefix = "$" if actual else "~$"
    if amount == _ZERO:
        return "$0.00"
    if amount < _SUBCENT_THRESHOLD:
        label = f"{prefix}{amount:.4f}"
        zero_4dp = f"{prefix}0.0000"
        return label if label != zero_4dp else f"{prefix}<0.0001"
    return f"{prefix}{amount:.2f}"


def _extract_token_int(obj: Any, attr: str) -> int | None:
    """Read a non-negative int token counter from *obj*, or ``None`` if absent."""
    val = getattr(obj, attr, None)
    if val is None or isinstance(val, bool):
        return None
    try:
        parsed = int(val)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def estimate_cost_from_usage(
    model: str | None,
    usage: Any,
    *,
    entry: dict[str, Any] | None = None,
) -> Any | None:
    """Compute a ``CostResult`` for an Experiential Labs turn, or ``None``.

    Resolution order:
    1. ``:free`` promotional variants -> $0 ``included`` ("free allowance").
    2. Inline ``raw_usage`` BYOK marker (``is_byok: true``) -> upstream estimate
       when ``cost_details.upstream_inference_cost`` is present, else $0 BYOK.
    3. Inline ``raw_usage["cost"]`` (exact USD stamped by Experiential Labs on
       every non-idempotent response) -> ``actual`` via ``provider_cost_api``.
    4. Cached ``/v1/models`` nano-USD rate card (including cache read/write,
       1h cache writes, reasoning rate, long-context tier, and ``:fast`` 2x) ->
       ``estimated`` via ``provider_models_api``.
    5. Otherwise ``None`` (fail open to Hermes built-in path).
    """
    if not isinstance(model, str) or not model.strip():
        return None

    try:
        from agent.usage_pricing import CostResult
    except Exception:  # noqa: BLE001 - cost layer must fail open
        return None

    _, suffix = split_model_suffix(model)
    if suffix == ":free":
        return CostResult(
            amount_usd=_ZERO,
            status="included",
            source="none",
            label="free allowance",
            notes=(_FREE_ALLOWANCE_NOTE,),
        )

    raw = getattr(usage, "raw_usage", None)
    if isinstance(raw, dict):
        if raw.get("is_byok") is True:
            cost_details = raw.get("cost_details")
            upstream = (
                _to_decimal(cost_details.get("upstream_inference_cost"))
                if isinstance(cost_details, dict)
                else None
            )
            if upstream is not None and upstream >= _ZERO:
                return CostResult(
                    amount_usd=upstream,
                    status="estimated",
                    source="provider_cost_api",
                    label=_format_cost_label(upstream, actual=False),
                    notes=(_BYOK_ESTIMATE_NOTE,),
                )
            return CostResult(
                amount_usd=_ZERO,
                status="included",
                source="provider_cost_api",
                label="$0.00",
                notes=(_BYOK_ZERO_NOTE,),
            )

        inline_cost = _to_decimal(raw.get("cost"))
        if inline_cost is not None and inline_cost >= _ZERO:
            return CostResult(
                amount_usd=inline_cost,
                status="actual",
                source="provider_cost_api",
                label=_format_cost_label(inline_cost, actual=True),
                notes=(),
            )

    # Rate-card fallback from cached /v1/models pricing.
    if not isinstance(entry, dict):
        return None
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return None

    in_tok = _extract_token_int(usage, "input_tokens")
    out_tok = _extract_token_int(usage, "output_tokens")
    if in_tok is None and out_tok is None:
        return None
    in_tok = in_tok or 0
    out_tok = out_tok or 0
    cache_read_tok = _extract_token_int(usage, "cache_read_tokens") or 0
    cache_write_tok = _extract_token_int(usage, "cache_write_tokens") or 0
    reasoning_tok = _extract_token_int(usage, "reasoning_tokens") or 0

    prompt_total = in_tok + cache_read_tok + cache_write_tok
    tier = pricing
    lc = pricing.get("long_context")
    if isinstance(lc, dict):
        threshold = _to_positive_int(lc.get("threshold"))
        if threshold and prompt_total > threshold:
            tier = {**pricing, **lc}

    in_rate = _to_decimal(tier.get("input"))
    out_rate = _to_decimal(tier.get("output"))
    if in_rate is None or out_rate is None:
        return None

    cached_rate = _to_decimal(tier.get("cached_input"))
    if cached_rate is None:
        cached_rate = in_rate
    write_rate = _to_decimal(tier.get("cache_write"))
    if write_rate is None:
        write_rate = in_rate
    write_1h_rate = _to_decimal(tier.get("cache_write_1h")) or write_rate
    reasoning_rate = _to_decimal(tier.get("reasoning"))

    amount = _ZERO
    amount += Decimal(in_tok) * in_rate / _NANO_PER_MILLION_DIVISOR
    amount += Decimal(cache_read_tok) * cached_rate / _NANO_PER_MILLION_DIVISOR

    # Cache writes: honour per-TTL breakdown when reported (e.g. 1h writes).
    by_ttl = getattr(usage, "cache_write_by_ttl", None)
    if isinstance(by_ttl, (list, tuple)) and by_ttl:
        accounted_write = 0
        for ttl_label, ttl_tokens in by_ttl:
            tok = _to_positive_int(ttl_tokens)
            if not tok:
                continue
            accounted_write += tok
            rate = write_1h_rate if str(ttl_label).strip().lower() == "1h" else write_rate
            amount += Decimal(tok) * rate / _NANO_PER_MILLION_DIVISOR
        leftover_write = max(0, cache_write_tok - accounted_write)
        if leftover_write:
            amount += Decimal(leftover_write) * write_rate / _NANO_PER_MILLION_DIVISOR
    elif cache_write_tok:
        amount += Decimal(cache_write_tok) * write_rate / _NANO_PER_MILLION_DIVISOR

    if reasoning_rate is not None and reasoning_tok:
        # When a model bills reasoning tokens at a separate rate, bill non-reasoning
        # completion tokens at output_rate and reasoning tokens at reasoning_rate.
        text_out = max(0, out_tok - reasoning_tok)
        amount += Decimal(text_out) * out_rate / _NANO_PER_MILLION_DIVISOR
        amount += Decimal(reasoning_tok) * reasoning_rate / _NANO_PER_MILLION_DIVISOR
    else:
        amount += Decimal(out_tok) * out_rate / _NANO_PER_MILLION_DIVISOR

    if suffix == ":fast":
        amount *= _FAST_MULTIPLIER

    return CostResult(
        amount_usd=amount,
        status="estimated",
        source="provider_models_api",
        label=_format_cost_label(amount, actual=False),
        notes=(_MODELS_API_ESTIMATE_NOTE,),
    )


# ── Per-home cache, disk mirror & async warmer ─────────────────────────


@dataclass
class _HomeCache:
    entries: dict[str, dict[str, Any]] | None = None
    failed_at: float | None = None
    disk_checked: bool = False
    warm_started: bool = False
    detail_contexts: dict[str, int] = field(default_factory=dict)


_CACHE_LOCK = threading.Lock()
_CACHES_BY_HOME: dict[str, _HomeCache] = {}


def _hermes_home_dir() -> Path | None:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:  # noqa: BLE001 - fail open outside Hermes
        return None


def _home_key() -> str:
    home = _hermes_home_dir()
    return str(home) if home is not None else "default"


def _state() -> _HomeCache:
    key = _home_key()
    with _CACHE_LOCK:
        cache = _CACHES_BY_HOME.get(key)
        if cache is None:
            cache = _HomeCache()
            _CACHES_BY_HOME[key] = cache
        return cache


def invalidate_catalog_cache() -> None:
    """Clear in-process catalog caches for the active Hermes home."""
    key = _home_key()
    with _CACHE_LOCK:
        _CACHES_BY_HOME.pop(key, None)


def _disk_cache_path() -> Path | None:
    home = _hermes_home_dir()
    if home is None:
        return None
    return home / "cache" / _DISK_CACHE_FILENAME


def _load_disk_cache() -> tuple[dict[str, dict[str, Any]] | None, float]:
    path = _disk_cache_path()
    if path is None or not path.exists():
        return None, 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None, 0.0
    if not isinstance(payload, dict):
        return None, 0.0
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        return None, 0.0
    try:
        age = max(0.0, time.time() - float(payload.get("ts") or 0))
    except (TypeError, ValueError):
        age = float(_DISK_TTL_S)
    clean = {str(k): v for k, v in entries.items() if isinstance(v, dict)}
    return (clean or None), age


def _save_disk_cache(entries: dict[str, dict[str, Any]]) -> None:
    path = _disk_cache_path()
    if path is None or not entries:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"ts": time.time(), "entries": entries}, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - best-effort disk mirror
        pass


def seed_catalog_from_v1_items(
    items: Any, *, save_disk: bool = True
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Parse ``/v1/models`` items, seed memory + disk cache, return ``(ids, entries)``."""
    ids, entries = parse_v1_catalog(items)
    if entries:
        st = _state()
        st.entries = entries
        st.failed_at = None
        st.disk_checked = True
        if save_disk:
            _save_disk_cache(entries)
    return ids, entries


def _resolve_credentials() -> tuple[str, str]:
    """Resolve ``(api_key, base_url)`` for background warming, fail-open."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        rt = resolve_runtime_provider(requested="experientiallabs")
        if isinstance(rt, dict):
            key = str(rt.get("api_key") or "").strip()
            base = str(rt.get("base_url") or "").strip().rstrip("/") or _DEFAULT_BASE_URL
            return key, base
    except Exception:  # noqa: BLE001
        pass
    key = (
        os.environ.get("EXPLABS_API_KEY")
        or os.environ.get("EXPERIENTIALLABS_API_KEY")
        or ""
    ).strip()
    base = (os.environ.get("EXPLABS_BASE_URL") or "").strip().rstrip("/") or _DEFAULT_BASE_URL
    return key, base


def _models_endpoint(base_url: str | None) -> str:
    base = (base_url or "").strip().rstrip("/") or _DEFAULT_BASE_URL
    return f"{base}/models"


def _user_agent() -> str:
    try:
        from providers.base import _profile_user_agent

        return _profile_user_agent()
    except Exception:  # noqa: BLE001
        return "hermes-cli"


def fetch_v1_catalog(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 8.0,
) -> list[str] | None:
    """Fetch ``GET /v1/models``, seed the catalog cache, return chat/tool model IDs.

    Returns ``None`` on any failure so callers fall back cleanly.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None

    headers = {"Accept": "application/json", "User-Agent": _user_agent()}
    token = (api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = _models_endpoint(base_url)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
        if response.status_code != 200:
            _state().failed_at = time.monotonic()
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001
        _state().failed_at = time.monotonic()
        return None

    items = payload if isinstance(payload, list) else (
        payload.get("data") if isinstance(payload, dict) else None
    )
    if not isinstance(items, list):
        _state().failed_at = time.monotonic()
        return None

    ids, entries = seed_catalog_from_v1_items(items)
    return ids if ids else (list(entries.keys()) or None)


def _warm_catalog_sync() -> None:
    st = _state()
    if st.failed_at is not None and (time.monotonic() - st.failed_at) < _FAIL_COOLDOWN_S:
        return
    api_key, base_url = _resolve_credentials()
    if not api_key:
        return
    fetch_v1_catalog(api_key=api_key, base_url=base_url, timeout=6.0)


def warm_catalog_async() -> None:
    """Fire-and-forget background warm of ``/v1/models`` (skipped under pytest)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    st = _state()
    if st.warm_started:
        return
    st.warm_started = True
    try:
        ctx = contextvars.copy_context()
        threading.Thread(
            target=ctx.run,
            args=(_warm_catalog_sync,),
            name="explabs-catalog-warm",
            daemon=True,
        ).start()
    except Exception:  # noqa: BLE001
        pass


def get_cached_entries(*, warm: bool = True) -> dict[str, dict[str, Any]] | None:
    """Cache-only map of ``model_id -> entry`` (memory, then disk mirror; never HTTP)."""
    st = _state()
    if st.entries is None and not st.disk_checked:
        st.disk_checked = True
        disk_entries, age = _load_disk_cache()
        if disk_entries is not None:
            st.entries = disk_entries
            if warm and age >= _DISK_TTL_S:
                warm_catalog_async()
    if st.entries is None and warm:
        warm_catalog_async()
    return st.entries


def get_model_entry(model: str | None, *, warm: bool = True) -> dict[str, Any] | None:
    """Cache-only metadata entry for *model* (resolving suffixes / vendor prefix)."""
    entries = get_cached_entries(warm=warm)
    if not entries:
        return None
    for candidate in _lookup_candidates(model):
        found = entries.get(candidate)
        if found is not None:
            return found
    # Case-insensitive fallback if exact case missed.
    lowered = {k.lower(): v for k, v in entries.items()}
    for candidate in _lookup_candidates(model):
        found = lowered.get(candidate.lower())
        if found is not None:
            return found
    return None


def fetch_model_context_live(
    model: str | None,
    *,
    api_key: str = "",
    base_url: str = "",
    timeout: float = _LIVE_DETAIL_TIMEOUT_S,
) -> int | None:
    """Live context-window lookup via keyless ``GET /api/models/<slug>``.

    Used by ``ProviderProfile.live_context_length`` when the cache missed.
    Returns ``max_input_tokens`` (preferred) or ``context_window``, else ``None``.
    """
    slug = base_model_slug(model)
    if "/" in slug:
        slug = slug.rsplit("/", 1)[-1].strip()
    if not slug:
        return None

    st = _state()
    if slug in st.detail_contexts:
        return st.detail_contexts[slug]

    entry = get_model_entry(model, warm=False)
    if entry and isinstance(entry.get("context_window"), int):
        return entry["context_window"]

    try:
        import httpx
        from urllib.parse import quote
    except Exception:  # noqa: BLE001
        return None

    url = f"{_PUBLIC_MODEL_DETAIL_BASE}/{quote(slug, safe='')}"
    headers = {"Accept": "application/json", "User-Agent": _user_agent()}
    token = (api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
        if response.status_code != 200:
            return None
        ctx = parse_public_model_detail_context(response.json())
    except Exception:  # noqa: BLE001
        return None

    if ctx is not None:
        st.detail_contexts[slug] = ctx
    return ctx
