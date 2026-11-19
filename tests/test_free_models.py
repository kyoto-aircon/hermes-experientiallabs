"""Tests for dynamic ``:free`` model discovery and allowance crediting.

The gateway's ``:free`` variants are promotional routes drawn from a separate
free allowance (not wallet credits), and they come and go with promotions.
These tests pin: the promotions→slug builder (against the live-captured
payload shape), the picker merge, the fail-open network probe, the profile's
``fetch_models`` override, and the $0 ``get_usage_cost`` crediting split.
Everything is offline; the real-gateway probe is ``integration``-marked.
"""

from __future__ import annotations

import sys
import types
from decimal import Decimal

import pytest

# Trimmed capture of the live public catalog's ``promotions`` array
# (GET /api/models?limit=1, 2026-11-19): an allowance-style free promo, a
# time-boxed "free" badge promo with USD caps, and a plain 50% discount.
LIVE_PROMOTIONS = [
    {
        "label": "GPT-6 Decisions API: free until Nov 14",
        "display_only": False,
        "slugs": ["gpt-6-luna-decisions"],
        "listed_slugs": [],
        "free": True,
        "badge_style": "allowance",
        "free_only_via_suffix": True,
        "percent_off": 0.0,
        "free_input_tokens_per_day": 5999994,
        "free_output_tokens_per_day": 1999998,
        "free_input_tokens_per_hour": 1999998,
        "free_output_tokens_per_hour": 666666,
        "requires_payment_method": True,
    },
    {
        "label": "GPT-6.2 Astra: 48 hours free",
        "display_only": False,
        "slugs": ["gpt-6.2-astra"],
        "listed_slugs": ["gpt-6.2-astra"],
        "free": False,
        "badge_style": "free",
        "free_only_via_suffix": True,
        "percent_off": 0.0,
        "per_org_hourly_cap_nano_usd": 12000000000,
        "model_terms": [{"slug": "gpt-6.2-astra", "max_daily_nano_usd": 36000000000}],
    },
    {
        "label": "TypeSafe Jev 50% off",
        "display_only": False,
        "slugs": ["jev-latest"],
        "listed_slugs": ["jev-latest"],
        "free": False,
        "badge_style": "allowance",
        "free_only_via_suffix": False,
        "percent_off": 50.0,
    },
]


@pytest.fixture(scope="module")
def free_models(plugin):
    import importlib

    return importlib.import_module(f"{plugin.__name__}.free_models")


class TestFreeSuffixSlugs:
    def test_live_payload_yields_both_free_variants(self, free_models):
        assert free_models.free_suffix_slugs(LIVE_PROMOTIONS) == (
            "gpt-6-luna-decisions:free",
            "gpt-6.2-astra:free",
        )

    def test_discount_promo_produces_no_variant(self, free_models):
        assert free_models.free_suffix_slugs([LIVE_PROMOTIONS[2]]) == ()

    def test_display_only_promo_is_skipped(self, free_models):
        promo = {**LIVE_PROMOTIONS[0], "display_only": True}
        assert free_models.free_suffix_slugs([promo]) == ()

    def test_free_without_suffix_gate_produces_no_variant(self, free_models):
        # free=True but free_only_via_suffix=False: the base slug itself is
        # free; there is no suffix route to advertise.
        promo = {**LIVE_PROMOTIONS[0], "free_only_via_suffix": False}
        assert free_models.free_suffix_slugs([promo]) == ()

    def test_positive_token_allowance_qualifies_without_flags(self, free_models):
        promo = {
            "slugs": ["allowance-model"],
            "free_only_via_suffix": True,
            "free_output_tokens_per_day": 1000,
        }
        assert free_models.free_suffix_slugs([promo]) == ("allowance-model:free",)

    def test_full_discount_qualifies(self, free_models):
        promo = {
            "slugs": ["hundred-off"],
            "free_only_via_suffix": True,
            "percent_off": 100.0,
        }
        assert free_models.free_suffix_slugs([promo]) == ("hundred-off:free",)

    def test_already_suffixed_slug_is_not_double_suffixed(self, free_models):
        promo = {
            "slugs": ["already:free", "plain"],
            "free_only_via_suffix": True,
            "free": True,
        }
        assert free_models.free_suffix_slugs([promo]) == ("plain:free",)

    def test_dedupes_across_slug_lists_and_promotions(self, free_models):
        promo = {
            "slugs": ["dup", "dup"],
            "listed_slugs": ["dup", "other"],
            "free_only_via_suffix": True,
            "free": True,
        }
        assert free_models.free_suffix_slugs([promo, promo]) == (
            "dup:free",
            "other:free",
        )

    @pytest.mark.parametrize(
        "payload",
        [None, {}, "not-a-list", 42, [None, 7, "x"], [{"slugs": "not-a-list"}]],
    )
    def test_malformed_payloads_are_tolerated(self, free_models, payload):
        assert free_models.free_suffix_slugs(payload) == ()

    def test_non_string_and_blank_slugs_are_skipped(self, free_models):
        promo = {
            "slugs": [None, 5, "", "   ", "ok"],
            "free_only_via_suffix": True,
            "free": True,
        }
        assert free_models.free_suffix_slugs([promo]) == ("ok:free",)


class TestMergeFreeSlugs:
    def test_no_free_variants_keeps_live_unchanged(self, free_models):
        live = ["a", "b"]
        assert free_models.merge_free_slugs(live, ()) is live
        assert free_models.merge_free_slugs(None, ()) is None

    def test_appends_variants_to_live_list(self, free_models):
        assert free_models.merge_free_slugs(["a", "b"], ("c:free",)) == [
            "a",
            "b",
            "c:free",
        ]

    def test_deduplicates_case_insensitively(self, free_models):
        assert free_models.merge_free_slugs(
            ["a", "B:free"], ("b:free", "c:free")
        ) == ["a", "B:free", "c:free"]

    def test_variants_alone_when_live_failed(self, free_models):
        assert free_models.merge_free_slugs(None, ("c:free",)) == ["c:free"]
        assert free_models.merge_free_slugs([], ("c:free",)) == ["c:free"]

    def test_neither_source_yields_none(self, free_models):
        assert free_models.merge_free_slugs([], ()) is None


def _fake_httpx(
    monkeypatch,
    *,
    status=200,
    payload=None,
    json_error=False,
    get_error=False,
    captured=None,
):
    """Install a stub ``httpx`` module (the fetch imports it lazily)."""
    mod = types.ModuleType("httpx")

    class _Response:
        status_code = status

        def json(self):
            if json_error:
                raise ValueError("bad json")
            return payload

    class _Client:
        def __init__(self, **kwargs):
            if captured is not None:
                captured["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers=None):
            if captured is not None:
                captured["url"] = url
                captured["headers"] = headers
            if get_error:
                raise OSError("network down")
            return _Response()

    mod.Client = _Client
    monkeypatch.setitem(sys.modules, "httpx", mod)
    return mod


class TestFetchFreeSuffixSlugs:
    def test_success_uses_the_cheap_keyless_probe(self, free_models, monkeypatch):
        captured = {}
        _fake_httpx(
            monkeypatch,
            payload={"promotions": LIVE_PROMOTIONS},
            captured=captured,
        )
        assert free_models.fetch_free_suffix_slugs() == (
            "gpt-6-luna-decisions:free",
            "gpt-6.2-astra:free",
        )
        assert captured["url"] == free_models._CATALOG_PROBE_URL
        assert "limit=1" in captured["url"]
        # Keyless: no Authorization header on the public catalog probe.
        assert "Authorization" not in (captured["headers"] or {})
        assert captured["client_kwargs"]["timeout"] == free_models._FREE_PROBE_TIMEOUT_S

    def test_timeout_is_forwarded(self, free_models, monkeypatch):
        captured = {}
        _fake_httpx(monkeypatch, payload={"promotions": []}, captured=captured)
        free_models.fetch_free_suffix_slugs(timeout=1.5)
        assert captured["client_kwargs"]["timeout"] == 1.5

    def test_non_200_fails_open(self, free_models, monkeypatch):
        _fake_httpx(monkeypatch, status=503, payload={"promotions": LIVE_PROMOTIONS})
        assert free_models.fetch_free_suffix_slugs() == ()

    def test_bad_json_fails_open(self, free_models, monkeypatch):
        _fake_httpx(monkeypatch, json_error=True)
        assert free_models.fetch_free_suffix_slugs() == ()

    def test_network_error_fails_open(self, free_models, monkeypatch):
        _fake_httpx(monkeypatch, get_error=True)
        assert free_models.fetch_free_suffix_slugs() == ()

    def test_missing_httpx_fails_open(self, free_models, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", None)
        assert free_models.fetch_free_suffix_slugs() == ()

    @pytest.mark.parametrize("payload", [None, [], {"models": []}, {"promotions": "x"}])
    def test_unexpected_shapes_fail_open(self, free_models, monkeypatch, payload):
        _fake_httpx(monkeypatch, payload=payload)
        assert free_models.fetch_free_suffix_slugs() == ()


class TestProfileFetchModels:
    """The override delegates to the base listing and merges the promo probe."""

    def _patch(self, monkeypatch, plugin, *, live, free):
        owner = sys.modules[type(plugin.experientiallabs).__module__]
        captured = {}

        def fake_catalog(**kwargs):
            captured.update(kwargs)
            return live

        def fake_probe(*, timeout=4.0):
            captured["probe_timeout"] = timeout
            return free

        monkeypatch.setattr(owner, "fetch_v1_catalog", fake_catalog)
        monkeypatch.setattr(owner, "fetch_free_suffix_slugs", fake_probe)
        return captured

    def test_merges_variants_into_the_live_list(self, plugin, monkeypatch):
        captured = self._patch(
            monkeypatch, plugin, live=["a", "b"], free=("b:free", "c:free")
        )
        result = plugin.experientiallabs.fetch_models(
            **{"api_key": "tok", "base_url": "https://api.experientiallabs.ai/v1"}
        )
        assert result == ["a", "b", "b:free", "c:free"]
        assert captured["api_key"] == "tok"
        assert captured["base_url"] == "https://api.experientiallabs.ai/v1"
        assert captured["timeout"] == 8.0
        # The probe is clamped to its own budget inside the picker's 8s.
        assert captured["probe_timeout"] == 4.0

    def test_probe_timeout_never_exceeds_the_caller_budget(self, plugin, monkeypatch):
        captured = self._patch(monkeypatch, plugin, live=["a"], free=())
        plugin.experientiallabs.fetch_models(timeout=2.0)
        assert captured["probe_timeout"] == 2.0

    def test_live_failure_still_offers_variants(self, plugin, monkeypatch):
        self._patch(monkeypatch, plugin, live=None, free=("c:free",))
        assert plugin.experientiallabs.fetch_models() == ["c:free"]

    def test_probe_failure_keeps_the_live_list(self, plugin, monkeypatch):
        self._patch(monkeypatch, plugin, live=["a", "b"], free=())
        assert plugin.experientiallabs.fetch_models() == ["a", "b"]

    def test_both_failed_falls_back_to_curated(self, plugin, monkeypatch):
        self._patch(monkeypatch, plugin, live=None, free=())
        assert plugin.experientiallabs.fetch_models() is None


class TestUsageCost:
    def test_free_variant_is_zero_cost_allowance(self, plugin):
        result = plugin.experientiallabs.get_usage_cost("gpt-6.2-astra:free", object())
        assert result is not None
        assert result.amount_usd == Decimal("0")
        assert result.status == "included"
        assert result.source == "none"
        assert result.label == "free allowance"
        assert result.notes  # explains the allowance-vs-wallet split

    @pytest.mark.parametrize(
        "model", ["gpt-6.2-astra", "gemma-4-26b-a4b-it", "", None, 42]
    )
    def test_paid_and_invalid_models_keep_the_builtin_path(self, plugin, model):
        assert plugin.experientiallabs.get_usage_cost(model, object()) is None

    def test_missing_cost_layer_fails_open(self, plugin, monkeypatch):
        monkeypatch.setitem(sys.modules, "agent.usage_pricing", None)
        assert plugin.experientiallabs.get_usage_cost("x:free", object()) is None

    def test_hermes_cost_pipeline_charges_zero_for_free_variants(
        self, plugin, monkeypatch
    ):
        import providers
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        # Point the pipeline at this repo's profile even if an installed copy
        # shadows provider discovery (same guard as test_classify.py).
        monkeypatch.setattr(
            providers,
            "get_provider_profile",
            lambda name: plugin.experientiallabs if name == "experientiallabs" else None,
        )
        usage = CanonicalUsage(input_tokens=1000, output_tokens=500)
        result = estimate_usage_cost(
            "gpt-6.2-astra:free", usage, provider="experientiallabs"
        )
        assert result.amount_usd == Decimal("0")
        assert result.status == "included"
        assert result.label == "free allowance"


@pytest.mark.integration
class TestLiveCatalogProbe:
    def test_real_probe_returns_slugs_without_a_key(self, free_models):
        # Keyless by design; an empty tuple is a valid answer (no active
        # free promotions), a raise is not.
        result = free_models.fetch_free_suffix_slugs()
        assert isinstance(result, tuple)
        assert all(slug.endswith(":free") for slug in result)
