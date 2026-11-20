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
import math
import os
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

_DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
_PUBLIC_MODEL_DETAIL_BASE = "https://api.experientiallabs.ai/api/models"
_PUBLIC_CATALOG_URL = "https://api.experientiallabs.ai/api/models?limit=1000"
_DISK_CACHE_FILENAME = "experientiallabs_catalog.json"
_DISK_TTL_S = 24 * 3600
_FAIL_COOLDOWN_S = 60.0
_LIVE_DETAIL_TIMEOUT_S = 5.0
_PUBLIC_CATALOG_TIMEOUT_S = 8.0

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


def _to_temperature(value: Any) -> float | None:
    """Parse a finite temperature bound (``bool``/NaN/inf rejected; ``0.0`` valid)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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

    for flag in (
        "supports_tools",
        "supports_completions",
        "supports_reasoning",
        "supports_temperature",
        "sampling_requires_reasoning_none",
    ):
        val = item.get(flag)
        if isinstance(val, bool):
            entry[flag] = val

    # Sampling constraints: pinned routes publish minimum == maximum (Claude 5
    # family pins 1.0); ranged routes publish an accepted interval. Absent or
    # malformed bounds stay absent so the clamp fails open.
    for bound in ("minimum_temperature", "maximum_temperature"):
        val = _to_temperature(item.get(bound))
        if val is not None:
            entry[bound] = val

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


def parse_public_catalog_meta(payload: Any) -> dict[str, dict[str, Any]]:
    """Extract per-slug vision/modalities, ``max_input_tokens``, and maker from
    ``GET /api/models`` (keyless public catalog).

    ``GET /v1/models`` omits ``input_modalities``, ``max_input_tokens``, and
    ``maker``; this enrichment joins them onto cached entries so Hermes'
    ``get_model_capabilities()`` has accurate per-model vision, input budget,
    and model family.
    """
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("models")
    if not isinstance(rows, list):
        return {}
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_obj = row.get("model")
        if not isinstance(model_obj, dict):
            continue
        slug = str(model_obj.get("slug") or "").strip()
        if not slug:
            continue
        patch: dict[str, Any] = {}
        mods = model_obj.get("input_modalities")
        if isinstance(mods, list):
            norm_mods = [str(m).strip().lower() for m in mods if isinstance(m, str)]
            patch["supports_vision"] = "image" in norm_mods
            if norm_mods:
                patch["input_modalities"] = norm_mods
        max_in = _to_positive_int(model_obj.get("max_input_tokens"))
        if max_in is not None:
            patch["max_input_tokens"] = max_in
        ctx = _to_positive_int(model_obj.get("context_window"))
        if ctx is not None:
            patch["context_window"] = ctx
        max_out = _to_positive_int(model_obj.get("max_output_tokens"))
        if max_out is not None:
            patch["max_output_tokens"] = max_out
        maker = model_obj.get("maker")
        if isinstance(maker, str) and maker.strip():
            patch["model_family"] = maker.strip()
        params = model_obj.get("supported_params")
        if isinstance(params, dict):
            if isinstance(params.get("tools"), bool):
                patch["supports_tools"] = params["tools"]
            if isinstance(params.get("reasoning"), bool):
                patch["supports_reasoning"] = params["reasoning"]
        if patch:
            meta[slug] = patch
    return meta


def merge_public_meta_into_entries(
    entries: dict[str, dict[str, Any]],
    public_meta: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Enrich ``/v1/models`` entries with keyless ``/api/models`` metadata.

    ``/v1/models`` wins on ``supports_tools`` / ``supports_reasoning`` (it
    reflects the org's actual routable lanes); ``/api/models`` supplies
    ``supports_vision``, ``input_modalities``, ``model_family``, and
    ``max_input_tokens`` (which overrides raw ``context_window`` when tighter).
    Public-only slugs not yet in ``entries`` are added too so capability
    lookups work even before a keyed ``/v1/models`` call.
    """
    merged: dict[str, dict[str, Any]] = {k: dict(v) for k, v in entries.items()}
    for slug, patch in public_meta.items():
        cur = merged.get(slug)
        if cur is None:
            entry: dict[str, Any] = {"id": slug}
            if "supports_vision" in patch:
                entry["supports_vision"] = patch["supports_vision"]
            if "input_modalities" in patch:
                entry["input_modalities"] = patch["input_modalities"]
            if "max_input_tokens" in patch:
                entry["context_window"] = patch["max_input_tokens"]
            elif "context_window" in patch:
                entry["context_window"] = patch["context_window"]
            if "max_output_tokens" in patch:
                entry["max_output_tokens"] = patch["max_output_tokens"]
            if "model_family" in patch:
                entry["model_family"] = patch["model_family"]
            if "supports_tools" in patch:
                entry["supports_tools"] = patch["supports_tools"]
            if "supports_reasoning" in patch:
                entry["supports_reasoning"] = patch["supports_reasoning"]
            merged[slug] = entry
            continue
        if "supports_vision" in patch:
            cur["supports_vision"] = patch["supports_vision"]
        if "input_modalities" in patch:
            cur["input_modalities"] = patch["input_modalities"]
        if "model_family" in patch and not cur.get("model_family"):
            cur["model_family"] = patch["model_family"]
        if "max_input_tokens" in patch:
            # Gateway docs: max_input_tokens is the real input budget when a
            # maker counts max_output inside context_window.
            cur["context_window"] = patch["max_input_tokens"]
        elif "context_window" in patch and "context_window" not in cur:
            cur["context_window"] = patch["context_window"]
        if "max_output_tokens" in patch and "max_output_tokens" not in cur:
            cur["max_output_tokens"] = patch["max_output_tokens"]
    return merged


def entry_to_model_capabilities(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Convert one cached catalog entry into Hermes' canonical
    ``ProviderProfile.model_capabilities`` schema (``agent/models_dev.py``).

    Emits only keys actually attested by the catalog:
    ``supports_tools``, ``supports_reasoning``, ``supports_vision``,
    ``context_window``, ``max_output_tokens``, ``model_family``.
    """
    if not isinstance(entry, dict):
        return {}
    caps: dict[str, Any] = {}
    if isinstance(entry.get("supports_tools"), bool):
        caps["supports_tools"] = entry["supports_tools"]
    if isinstance(entry.get("supports_reasoning"), bool):
        caps["supports_reasoning"] = entry["supports_reasoning"]
    if isinstance(entry.get("supports_vision"), bool):
        caps["supports_vision"] = entry["supports_vision"]
    ctx = _to_positive_int(entry.get("context_window"))
    if ctx is not None:
        caps["context_window"] = ctx
    max_out = _to_positive_int(entry.get("max_output_tokens"))
    if max_out is not None:
        caps["max_output_tokens"] = max_out
    family = entry.get("model_family")
    if isinstance(family, str) and family.strip():
        caps["model_family"] = family.strip()
    return caps


class CatalogCapabilitiesView(dict):
    """Dynamic ``dict``-compatible view exposing live catalog capabilities to
    Hermes' ``ProviderProfile.model_capabilities`` seam.

    Hermes reads ``profile.model_capabilities.get(model, {})`` on hot paths
    (``agent/models_dev.py``) and expects a ``dict``. Subclassing ``dict``
    keeps ``isinstance(..., dict)`` true and zero-I/O on hot paths while
    answering exact slugs, ``:free``/``:fast``/``:online`` variants, and
    ``vendor/slug`` spellings from the memory/disk cache.
    """

    def get(self, key: Any, default: Any = None) -> Any:
        if not isinstance(key, str) or not key.strip():
            return default
        entry = get_model_entry(key, warm=True)
        caps = entry_to_model_capabilities(entry)
        return caps if caps else default

    def __getitem__(self, key: str) -> dict[str, Any]:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and bool(self.get(key))

    def _snapshot(self) -> dict[str, dict[str, Any]]:
        entries = get_cached_entries(warm=False) or {}
        out: dict[str, dict[str, Any]] = {}
        for slug, entry in entries.items():
            caps = entry_to_model_capabilities(entry)
            if caps:
                out[slug] = caps
        return out

    def __iter__(self):
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())

    def keys(self):
        return self._snapshot().keys()

    def values(self):
        return self._snapshot().values()

    def items(self):
        return self._snapshot().items()

    def copy(self) -> dict[str, dict[str, Any]]:
        return self._snapshot()

    def __bool__(self) -> bool:
        return bool(get_cached_entries(warm=False))


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
    public_meta_loaded: bool = False
    public_meta_warm_started: bool = False
    public_meta_failed_at: float | None = None
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
        # Preserve any already-merged public metadata (supports_vision,
        # model_family, tighter max_input_tokens) across v1 refreshes.
        existing = st.entries or {}
        if existing:
            for slug, new_entry in entries.items():
                prev = existing.get(slug)
                if not isinstance(prev, dict):
                    continue
                for carry in ("supports_vision", "input_modalities", "model_family"):
                    if carry in prev and carry not in new_entry:
                        new_entry[carry] = prev[carry]
                prev_ctx = _to_positive_int(prev.get("context_window"))
                new_ctx = _to_positive_int(new_entry.get("context_window"))
                if prev_ctx and new_ctx and prev_ctx < new_ctx and "input_modalities" in prev:
                    new_entry["context_window"] = prev_ctx
        st.entries = entries
        st.failed_at = None
        st.disk_checked = True
        if save_disk:
            _save_disk_cache(entries)
    return ids, entries


def seed_public_catalog_meta(
    payload: Any, *, save_disk: bool = True
) -> dict[str, dict[str, Any]]:
    """Parse keyless ``/api/models`` payload and merge vision/modality/budget
    metadata into the active home's cache."""
    meta = parse_public_catalog_meta(payload)
    if not meta:
        return {}
    st = _state()
    base = st.entries or {}
    merged = merge_public_meta_into_entries(base, meta)
    st.entries = merged
    st.public_meta_loaded = True
    st.public_meta_failed_at = None
    st.disk_checked = True
    if save_disk:
        _save_disk_cache(merged)
    return meta


def fetch_public_catalog_meta(
    *, timeout: float = _PUBLIC_CATALOG_TIMEOUT_S
) -> dict[str, dict[str, Any]]:
    """Fetch keyless ``GET /api/models?limit=1000`` and merge into catalog cache.

    Fails open (``{}``) on every error; intended for background warming so
    hot paths and ``/model`` pickers never pay its latency.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return {}
    headers = {"Accept": "application/json", "User-Agent": _user_agent()}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(_PUBLIC_CATALOG_URL, headers=headers)
        if response.status_code != 200:
            _state().public_meta_failed_at = time.monotonic()
            return {}
        payload = response.json()
    except Exception:  # noqa: BLE001
        _state().public_meta_failed_at = time.monotonic()
        return {}
    meta = seed_public_catalog_meta(payload)
    if not meta:
        _state().public_meta_failed_at = time.monotonic()
    return meta


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
    if entries and not _state().public_meta_loaded:
        warm_public_meta_async()
    return ids if ids else (list(entries.keys()) or None)


def _warm_catalog_sync() -> None:
    st = _state()
    if st.failed_at is not None and (time.monotonic() - st.failed_at) < _FAIL_COOLDOWN_S:
        return
    api_key, base_url = _resolve_credentials()
    if api_key:
        fetch_v1_catalog(api_key=api_key, base_url=base_url, timeout=6.0)
    if not st.public_meta_loaded:
        _warm_public_meta_sync()


def _warm_public_meta_sync() -> None:
    st = _state()
    if (
        st.public_meta_failed_at is not None
        and (time.monotonic() - st.public_meta_failed_at) < _FAIL_COOLDOWN_S
    ):
        return
    fetch_public_catalog_meta(timeout=_PUBLIC_CATALOG_TIMEOUT_S)


def warm_public_meta_async() -> None:
    """Background warm for keyless ``/api/models`` vision/modality enrichment."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    st = _state()
    if st.public_meta_warm_started or st.public_meta_loaded:
        return
    st.public_meta_warm_started = True
    try:
        ctx = contextvars.copy_context()
        threading.Thread(
            target=ctx.run,
            args=(_warm_public_meta_sync,),
            name="explabs-public-meta-warm",
            daemon=True,
        ).start()
    except Exception:  # noqa: BLE001
        pass


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
            st.public_meta_loaded = any(
                isinstance(e, dict) and "supports_vision" in e
                for e in disk_entries.values()
            )
            if warm and age >= _DISK_TTL_S:
                warm_catalog_async()
            elif warm and not st.public_meta_loaded:
                warm_public_meta_async()
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
