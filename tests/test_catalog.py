"""Tests for catalog parsing, reasoning caps, context bounds, cost attribution,
ZDR data_use, session/tag headers, and dynamic aux resolution.

All network calls are stubbed; real-gateway shapes match live captures from
``GET /v1/models`` and ``GET /api/models/<slug>``.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from decimal import Decimal

import pytest

# Live-captured shapes from Experiential Labs GET /v1/models (2026-11-20).
LIVE_V1_MODELS = [
    {
        "id": "claude-opus-5.5",
        "object": "model",
        "supports_completions": True,
        "supports_tools": True,
        "supports_reasoning": True,
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
        "reasoning_effort": "medium",
        "chat_max_tokens_field": None,
        "context_window_tokens": 1000000,
        "maximum_output_tokens": 128000,
        "pricing": {
            "input_nano_usd_per_million_tokens": 4000000000,
            "cached_input_nano_usd_per_million_tokens": 200000000,
            "cache_creation_input_nano_usd_per_million_tokens": 5000000000,
            "cache_creation_1h_input_nano_usd_per_million_tokens": 8000000000,
            "output_nano_usd_per_million_tokens": 20000000000,
            "reasoning_nano_usd_per_million_tokens": None,
            "long_context": None,
        },
    },
    {
        "id": "claude-haiku-4.5",
        "object": "model",
        "supports_completions": True,
        "supports_tools": True,
        "supports_reasoning": False,
        "supported_reasoning_efforts": [],
        "reasoning_effort": None,
        "chat_max_tokens_field": None,
        "context_window_tokens": 200000,
        "maximum_output_tokens": 64000,
        "pricing": {
            "input_nano_usd_per_million_tokens": 1000000000,
            "cached_input_nano_usd_per_million_tokens": 100000000,
            "output_nano_usd_per_million_tokens": 5000000000,
        },
    },
    {
        "id": "gpt-6-astra",
        "object": "model",
        "supports_completions": True,
        "supports_tools": True,
        "supports_reasoning": True,
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
        "reasoning_effort": "medium",
        "chat_max_tokens_field": "max_completion_tokens",
        "context_window_tokens": 1050000,
        "maximum_output_tokens": 128000,
        "pricing": {
            "input_nano_usd_per_million_tokens": 10000000000,
            "cached_input_nano_usd_per_million_tokens": 1000000000,
            "cache_creation_input_nano_usd_per_million_tokens": 12500000000,
            "output_nano_usd_per_million_tokens": 50000000000,
            "long_context": {
                "input_threshold_tokens": 272001,
                "input_nano_usd_per_million_tokens": 20000000000,
                "cached_input_nano_usd_per_million_tokens": 2000000000,
                "cache_creation_input_nano_usd_per_million_tokens": 25000000000,
                "output_nano_usd_per_million_tokens": 75000000000,
            },
        },
    },
    {
        "id": "deepseek-v4-flash",
        "object": "model",
        "supports_completions": True,
        "supports_tools": True,
        "supports_reasoning": True,
        "supported_reasoning_efforts": ["none", "high", "max"],
        "reasoning_effort": "none",
        "chat_max_tokens_field": "max_tokens",
        "context_window_tokens": 1048576,
        "maximum_output_tokens": 384000,
        "pricing": {
            "input_nano_usd_per_million_tokens": 190000000,
            "cached_input_nano_usd_per_million_tokens": 28000000,
            "output_nano_usd_per_million_tokens": 510000000,
        },
    },
    {
        "id": "eleven_flash_v2_5",
        "object": "model",
        "supports_completions": False,
        "supports_tools": False,
    },
    {
        "id": "codestral-embed",
        "object": "model",
        "supports_embeddings": True,
        "supports_completions": False,
        "supports_tools": False,
    },
    {
        "id": "aion-rp-llama-3.1-8b",
        "object": "model",
        "supports_completions": True,
        "supports_tools": False,
    },
]


@pytest.fixture(scope="module")
def catalog(plugin):
    return importlib.import_module(f"{plugin.__name__}.catalog")


@pytest.fixture(autouse=True)
def _clean_catalog_cache(catalog, monkeypatch, tmp_path):
    """Isolate every test's in-memory + disk catalog cache."""
    monkeypatch.setattr(catalog, "_hermes_home_dir", lambda: tmp_path)
    catalog.invalidate_catalog_cache()
    yield
    catalog.invalidate_catalog_cache()


class TestCatalogParsing:
    def test_filters_non_tool_and_non_completion_models(self, catalog):
        ids, entries = catalog.parse_v1_catalog(LIVE_V1_MODELS)
        assert ids == [
            "claude-opus-5.5",
            "claude-haiku-4.5",
            "gpt-6-astra",
            "deepseek-v4-flash",
        ]
        assert "eleven_flash_v2_5" not in ids
        assert "codestral-embed" not in ids
        assert "aion-rp-llama-3.1-8b" not in ids
        # Entries still keep metadata for all models in /v1/models.
        assert "claude-opus-5.5" in entries

    def test_parses_context_reasoning_max_tokens_and_pricing(self, catalog):
        _, entries = catalog.parse_v1_catalog(LIVE_V1_MODELS)
        opus = entries["claude-opus-5.5"]
        assert opus["context_window"] == 1000000
        assert opus["max_output_tokens"] == 128000
        assert opus["supports_reasoning"] is True
        assert opus["supported_reasoning_efforts"] == [
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        assert opus["default_reasoning_effort"] == "medium"
        assert opus["pricing"]["input"] == "4000000000"
        assert opus["pricing"]["cache_write_1h"] == "8000000000"

        astra = entries["gpt-6-astra"]
        assert astra["chat_max_tokens_field"] == "max_completion_tokens"
        assert astra["pricing"]["long_context"]["threshold"] == 272001

    def test_public_model_detail_prefers_max_input_tokens(self, catalog):
        payload = {
            "model": {
                "slug": "deepseek-v4.1-flash",
                "context_window": 1048576,
                "max_input_tokens": 655360,
            }
        }
        assert catalog.parse_public_model_detail_context(payload) == 655360

    def test_public_model_detail_falls_back_to_context_window(self, catalog):
        payload = {"model": {"context_window": 262144, "max_input_tokens": None}}
        assert catalog.parse_public_model_detail_context(payload) == 262144

    def test_suffix_and_vendor_lookup_resolves_base_entry(self, catalog):
        catalog.seed_catalog_from_v1_items(LIVE_V1_MODELS, save_disk=False)
        assert catalog.get_model_entry("claude-opus-5.5:free")["id"] == "claude-opus-5.5"
        assert catalog.get_model_entry("gpt-6-astra:fast")["id"] == "gpt-6-astra"
        assert (
            catalog.get_model_entry("anthropic/claude-opus-5.5:online")["id"]
            == "claude-opus-5.5"
        )


class TestReasoningClamping:
    def test_unset_reasoning_config_omits_field(self, catalog):
        _, entries = catalog.parse_v1_catalog(LIVE_V1_MODELS)
        assert (
            catalog.resolve_reasoning_effort(
                "claude-opus-5.5", None, entry=entries["claude-opus-5.5"]
            )
            is None
        )

    def test_mandatory_reasoning_omits_disable_instead_of_400(self, catalog):
        _, entries = catalog.parse_v1_catalog(LIVE_V1_MODELS)
        for cfg in ({"enabled": False}, {"effort": "none"}):
            assert (
                catalog.resolve_reasoning_effort(
                    "claude-opus-5.5", cfg, entry=entries["claude-opus-5.5"]
                )
                is None
            )

    def test_optional_reasoning_sends_none_when_disabled(self, catalog):
        _, entries = catalog.parse_v1_catalog(LIVE_V1_MODELS)
        for cfg in ({"enabled": False}, {"effort": "none"}):
            assert (
                catalog.resolve_reasoning_effort(
                    "deepseek-v4-flash", cfg, entry=entries["deepseek-v4-flash"]
                )
                == "none"
            )

    def test_non_reasoning_model_omits_all_efforts(self, catalog):
        _, entries = catalog.parse_v1_catalog(LIVE_V1_MODELS)
        for cfg in ({"enabled": True, "effort": "high"}, {"enabled": False}):
            assert (
                catalog.resolve_reasoning_effort(
                    "claude-haiku-4.5", cfg, entry=entries["claude-haiku-4.5"]
                )
                is None
            )

    def test_cold_cache_omits_disable_but_clamps_positive_effort(self, catalog):
        assert (
            catalog.resolve_reasoning_effort(
                "unknown-model", {"enabled": False}, entry=None
            )
            is None
        )
        assert (
            catalog.resolve_reasoning_effort(
                "unknown-model", {"enabled": True, "effort": "ultra"}, entry=None
            )
            == "max"
        )

    def test_clamps_to_nearest_weaker_supported_without_disabling(self, catalog):
        _, entries = catalog.parse_v1_catalog(LIVE_V1_MODELS)
        ds = entries["deepseek-v4-flash"]  # supported: ["none", "high", "max"]
        assert (
            catalog.resolve_reasoning_effort(
                "deepseek-v4-flash", {"enabled": True, "effort": "xhigh"}, entry=ds
            )
            == "high"
        )
        # "low" has no weaker positive level -> weakest positive ("high"), never "none".
        assert (
            catalog.resolve_reasoning_effort(
                "deepseek-v4-flash", {"enabled": True, "effort": "low"}, entry=ds
            )
            == "high"
        )


class TestCostAttribution:
    def test_inline_usage_cost_is_actual(self, plugin):
        from agent.usage_pricing import CanonicalUsage

        usage = CanonicalUsage(
            input_tokens=1000,
            output_tokens=200,
            raw_usage={"cost": 0.0125},
        )
        result = plugin.experientiallabs.get_usage_cost("claude-opus-5.5", usage)
        assert result is not None
        assert result.amount_usd == Decimal("0.0125")
        assert result.status == "actual"
        assert result.source == "provider_cost_api"
        assert result.label == "$0.01"

    def test_inline_subcent_cost_uses_4dp(self, plugin):
        from agent.usage_pricing import CanonicalUsage

        usage = CanonicalUsage(
            input_tokens=500,
            output_tokens=100,
            raw_usage={"cost": "0.0023"},
        )
        result = plugin.experientiallabs.get_usage_cost("deepseek-v4-flash", usage)
        assert result is not None
        assert result.amount_usd == Decimal("0.0023")
        assert result.status == "actual"
        assert result.label == "$0.0023"

    def test_byok_reports_upstream_estimate_or_zero(self, plugin):
        from agent.usage_pricing import CanonicalUsage

        with_est = CanonicalUsage(
            input_tokens=100,
            output_tokens=50,
            raw_usage={
                "cost": 0,
                "is_byok": True,
                "cost_details": {"upstream_inference_cost": "0.045"},
            },
        )
        r1 = plugin.experientiallabs.get_usage_cost("claude-opus-5.5", with_est)
        assert r1.amount_usd == Decimal("0.045")
        assert r1.status == "estimated"
        assert "BYOK" in r1.notes[0]

        no_est = CanonicalUsage(
            input_tokens=100,
            output_tokens=50,
            raw_usage={"cost": 0, "is_byok": True},
        )
        r2 = plugin.experientiallabs.get_usage_cost("claude-opus-5.5", no_est)
        assert r2.amount_usd == Decimal("0")
        assert r2.status == "included"
        assert "BYOK" in r2.notes[0]

    def test_rate_card_fallback_with_cache_and_1h_ttl(self, plugin, catalog):
        catalog.seed_catalog_from_v1_items(LIVE_V1_MODELS, save_disk=False)
        # claude-opus-5.5: $4/M in, $0.20/M cached, $5/M 5m write, $8/M 1h write, $20/M out.
        # SimpleNamespace keeps this test compatible with Hermes v0.21.6, where
        # CanonicalUsage did not yet declare cache_write_by_ttl.
        usage = types.SimpleNamespace(
            input_tokens=100_000,        # $0.40
            cache_read_tokens=500_000,   # $0.10
            cache_write_tokens=100_000,  # 50k 1h ($0.40) + 50k 5m ($0.25) = $0.65
            output_tokens=10_000,        # $0.20
            reasoning_tokens=0,
            raw_usage=None,
            cache_write_by_ttl=(("1h", 50_000), ("5m", 50_000)),
        )
        result = plugin.experientiallabs.get_usage_cost("claude-opus-5.5", usage)
        assert result is not None
        assert result.status == "estimated"
        assert result.source == "provider_models_api"
        assert result.amount_usd == Decimal("1.35")

    def test_rate_card_long_context_tier_and_fast_multiplier(self, plugin, catalog):
        from agent.usage_pricing import CanonicalUsage

        catalog.seed_catalog_from_v1_items(LIVE_V1_MODELS, save_disk=False)
        # gpt-6-astra long_context (>272001): $20/M in, $75/M out
        usage = CanonicalUsage(input_tokens=300_000, output_tokens=10_000)
        std = plugin.experientiallabs.get_usage_cost("gpt-6-astra", usage)
        # 0.3 * $20 + 0.01 * $75 = $6.00 + $0.75 = $6.75
        assert std.amount_usd == Decimal("6.75")

        fast = plugin.experientiallabs.get_usage_cost("gpt-6-astra:fast", usage)
        assert fast.amount_usd == Decimal("13.50")


class TestDiskMirrorAndLiveFetch:
    def test_disk_mirror_roundtrip(self, catalog):
        catalog.seed_catalog_from_v1_items(LIVE_V1_MODELS, save_disk=True)
        catalog.invalidate_catalog_cache()
        hydrated = catalog.get_model_entry("claude-opus-5.5", warm=False)
        assert hydrated is not None
        assert hydrated["context_window"] == 1000000

    def test_fetch_v1_catalog_seeds_cache_and_filters(self, catalog, monkeypatch):
        class FakeResp:
            status_code = 200

            def json(self):
                return {"object": "list", "data": LIVE_V1_MODELS}

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, headers=None):
                assert url == "https://api.experientiallabs.ai/v1/models"
                assert headers["Authorization"] == "Bearer test-key"
                return FakeResp()

        monkeypatch.setitem(
            sys.modules, "httpx", types.SimpleNamespace(Client=FakeClient)
        )
        ids = catalog.fetch_v1_catalog(api_key="test-key")
        assert ids == [
            "claude-opus-5.5",
            "claude-haiku-4.5",
            "gpt-6-astra",
            "deepseek-v4-flash",
        ]
        entry = catalog.get_model_entry("gpt-6-astra", warm=False)
        assert entry is not None
        assert entry["context_window"] == 1050000

    def test_live_context_length_queries_public_detail_when_cold(
        self, plugin, catalog, monkeypatch
    ):
        class FakeResp:
            status_code = 200

            def json(self):
                return {
                    "model": {
                        "slug": "deepseek-v4.1-flash",
                        "context_window": 1048576,
                        "max_input_tokens": 655360,
                    }
                }

        calls = []

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, headers=None):
                calls.append(url)
                return FakeResp()

        monkeypatch.setitem(
            sys.modules, "httpx", types.SimpleNamespace(Client=FakeClient)
        )
        ctx = plugin.experientiallabs.live_context_length("deepseek-v4.1-flash:free")
        assert ctx == 655360
        assert calls == ["https://api.experientiallabs.ai/api/models/deepseek-v4.1-flash"]
        # Second call is served from detail cache (no second HTTP call).
        assert plugin.experientiallabs.live_context_length("deepseek-v4.1-flash") == 655360
        assert len(calls) == 1


class TestProfileParityHooks:
    def test_build_extra_body_enforces_zdr_only_when_data_use_nonempty(self, plugin):
        assert plugin.experientiallabs.build_extra_body(session_id="s1") == {}
        assert (
            plugin.experientiallabs.build_extra_body(
                session_id="s1", data_use=frozenset()
            )
            == {}
        )
        assert plugin.experientiallabs.build_extra_body(
            session_id="s1", data_use=frozenset({"zero_retention"})
        ) == {"provider": {"zdr": True}}
        assert plugin.experientiallabs.build_extra_body(
            session_id="s1", data_use=frozenset({"no_training"})
        ) == {"provider": {"zdr": True}}

    def test_build_api_kwargs_extras_emits_headers_and_clamped_reasoning(
        self, plugin, catalog
    ):
        catalog.seed_catalog_from_v1_items(LIVE_V1_MODELS, save_disk=False)

        eb, top = plugin.experientiallabs.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            model="deepseek-v4-flash",
            session_id="sess_12345",
        )
        assert eb == {}
        assert top["reasoning_effort"] == "high"
        headers = top["extra_headers"]
        assert "X-Explabs-Session-Id" in headers
        assert headers["X-Explabs-Session-Id"]
        parsed_tags = json.loads(headers["X-Explabs-Tags"])
        assert parsed_tags["product"] == "hermes-agent"
        assert "conversation" not in parsed_tags

    def test_reasoning_payload_and_supported_efforts_and_max_tokens_param(
        self, plugin, catalog
    ):
        catalog.seed_catalog_from_v1_items(LIVE_V1_MODELS, save_disk=False)
        prof = plugin.experientiallabs

        # Host check
        assert (
            prof.reasoning_payload(
                "claude-opus-5.5", None, base_url="https://api.openai.com/v1"
            )
            is None
        )
        assert prof.reasoning_payload(
            "claude-opus-5.5", None, base_url="https://api.experientiallabs.ai/v1"
        ) == {"effort": "medium"}
        assert (
            prof.reasoning_payload(
                "claude-haiku-4.5", None, base_url="https://api.experientiallabs.ai/v1"
            )
            is None
        )
        assert prof.supported_reasoning_efforts("claude-opus-5.5") == (
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )
        assert prof.supported_reasoning_efforts("claude-haiku-4.5") == ()
        assert prof.supported_reasoning_efforts("not-in-catalog") is None

        assert prof.get_model_context_length("claude-opus-5.5") == 1000000
        assert prof.get_model_context_length("claude-opus-5.5:free") == 1000000
        assert (
            prof.max_tokens_param(
                base_url="https://api.experientiallabs.ai/v1", model="gpt-6-astra"
            )
            == "max_completion_tokens"
        )
        assert (
            prof.max_tokens_param(
                base_url="https://api.experientiallabs.ai/v1", model="deepseek-v4-flash"
            )
            == "max_tokens"
        )
        assert (
            prof.max_tokens_param(
                base_url="https://api.openai.com/v1", model="gpt-6-astra"
            )
            is None
        )

    def test_resolve_aux_model_tracks_live_catalog_when_warm(self, plugin, catalog):
        prof = plugin.experientiallabs
        # Cold cache -> static default
        assert prof.resolve_aux_model() == "glm-5.3-flash"
        assert prof.default_vision_model() == "glm-5.3-flash"

        # Warm catalog without glm-5.3-flash -> picks next live candidate
        catalog.seed_catalog_from_v1_items(LIVE_V1_MODELS, save_disk=False)
        assert prof.resolve_aux_model(vision=False) == "deepseek-v4-flash"
        assert prof.resolve_aux_model(vision=True) == "claude-haiku-4.5"
        assert prof.default_vision_model() == "claude-haiku-4.5"

    def test_optional_newer_profile_fields_applied_when_supported(self, plugin):
        from providers.base import ProviderProfile

        fields = getattr(ProviderProfile, "__dataclass_fields__", {})
        prof = plugin.experientiallabs
        if "hostnames" in fields:
            assert "api.experientiallabs.ai" in prof.hostnames
        if "aggregator" in fields:
            assert prof.aggregator is True
        if "catalog_aliases" in fields:
            assert prof.catalog_aliases is True
        if "replays_tool_call_signatures" in fields:
            assert prof.replays_tool_call_signatures is True
        if "setup_reasoning_efforts" in fields:
            assert callable(prof.setup_reasoning_efforts)
