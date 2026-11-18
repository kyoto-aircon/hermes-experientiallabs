"""Dynamic ``:free`` model discovery for the Experiential Labs gateway.

The gateway has two kinds of free access, credited differently:

- **Free models** — catalog routes priced at $0 (e.g. ``gemma-4-26b-a4b-it``).
  Stable; they bill $0 against the wallet / daily key cap like any other route.
- **``:free`` variants** — promotional routes (``<slug>:free``) that draw on a
  separate free allowance (per-promotion hourly/daily token quotas or USD
  caps), NOT wallet credits. They are dynamic: promotions appear and expire,
  so the set of valid ``:free`` slugs changes over time.

``:free`` variants are not catalog rows — the public catalog advertises them
through its ``promotions`` array (``free_only_via_suffix: true``). This module
derives the current suffix slugs from that array so the picker tracks the
gateway instead of a hardcoded list that goes stale. The probe is keyless and
cheap: ``GET /api/models?limit=1`` still returns the full ``promotions`` array
(~12 KB versus ~4 MB for the full catalog).

The pure builders (``free_suffix_slugs``, ``merge_free_slugs``) do no I/O and
stay unit-testable; only ``fetch_free_suffix_slugs`` talks to the network, and
it fails open (``()``) on any error — a missing promo probe must never break
model listing. Only stdlib is imported at module load; ``httpx`` is imported
lazily inside the fetch path.
"""

from __future__ import annotations

from typing import Any

# Keyless public catalog. ``limit=1`` keeps the response tiny while still
# carrying the complete ``promotions`` array (verified against the live API).
_CATALOG_PROBE_URL = "https://api.experientiallabs.ai/api/models?limit=1"
# The probe rides along with the picker's live /v1/models fetch (8s default);
# keep it short so a slow catalog cannot dominate the picker's latency.
_FREE_PROBE_TIMEOUT_S = 4.0

# Promotional allowance fields: a positive quota means the promo grants free
# tokens even when the ``free`` flag / badge are absent.
_ALLOWANCE_KEYS = (
    "free_input_tokens_per_day",
    "free_output_tokens_per_day",
    "free_input_tokens_per_hour",
    "free_output_tokens_per_hour",
)


def _offers_free_access(promo: dict) -> bool:
    """True when a promotion grants free access (not merely a discount).

    Observed shapes: ``free: true`` with token allowances (badge
    ``allowance``), and a time-boxed ``badge_style: "free"`` promo with
    ``free: false`` plus USD caps. A 100%-off or positive-token-allowance
    promo also qualifies; a partial discount (e.g. 50% off) does not.
    """
    if promo.get("free") is True:
        return True
    if str(promo.get("badge_style") or "").strip().lower() == "free":
        return True
    percent = promo.get("percent_off")
    if isinstance(percent, (int, float)) and not isinstance(percent, bool) and percent >= 100:
        return True
    for key in _ALLOWANCE_KEYS:
        value = promo.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return True
    return False


def free_suffix_slugs(promotions: Any) -> tuple[str, ...]:
    """``<slug>:free`` variants advertised by a catalog ``promotions`` array.

    Pure: tolerant of malformed entries, order-preserving, case-insensitively
    deduplicated. Only promotions that both grant free access and gate it
    behind the suffix (``free_only_via_suffix``) produce variants; display-only
    ads and plain discounts produce none. Slugs already carrying a ``:free``
    suffix are never double-suffixed.
    """
    if not isinstance(promotions, list):
        return ()
    slugs: list[str] = []
    seen: set[str] = set()
    for promo in promotions:
        if not isinstance(promo, dict):
            continue
        if promo.get("display_only") or not promo.get("free_only_via_suffix"):
            continue
        if not _offers_free_access(promo):
            continue
        for key in ("slugs", "listed_slugs"):
            raw = promo.get(key)
            if not isinstance(raw, list):
                continue
            for slug in raw:
                if not isinstance(slug, str):
                    continue
                slug = slug.strip()
                if not slug or slug.lower().endswith(":free"):
                    continue
                variant = slug + ":free"
                if variant.lower() not in seen:
                    seen.add(variant.lower())
                    slugs.append(variant)
    return tuple(slugs)


def merge_free_slugs(
    live: list[str] | None, free: tuple[str, ...]
) -> list[str] | None:
    """Append promo ``:free`` variants to a live model list, deduplicated.

    ``None``/empty semantics match the picker contract: no free variants →
    the live list unchanged (including ``None``); variants with no live list →
    the variants alone (the curated fallback still merges in upstream);
    neither → ``None`` so Hermes falls back to ``fallback_models``.
    """
    if not free:
        return live or None
    merged = list(live) if live else []
    seen = {str(model).lower() for model in merged}
    for slug in free:
        if slug.lower() not in seen:
            seen.add(slug.lower())
            merged.append(slug)
    return merged or None


def fetch_free_suffix_slugs(*, timeout: float = _FREE_PROBE_TIMEOUT_S) -> tuple[str, ...]:
    """Current ``:free`` variants from the keyless public catalog, or ``()``.

    Fails open on every error path (import, network, status, JSON, shape):
    the promo probe is an enhancement to model listing, never a dependency.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001 - optional dependency must fail open
        return ()
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(_CATALOG_PROBE_URL, headers={"Accept": "application/json"})
        if response.status_code != 200:
            return ()
        payload = response.json()
    except Exception:  # noqa: BLE001 - fail open
        return ()
    promotions = payload.get("promotions") if isinstance(payload, dict) else None
    return free_suffix_slugs(promotions)
