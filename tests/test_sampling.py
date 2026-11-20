"""Offline tests for W1 (sampling clamp): catalog parsing of per-model
temperature constraints, the pure override builder, the profile hook, and the
classify hook's decline on temperature rejections.

All catalog entries are fakes seeded in-process (no network). Shapes mirror
live ``GET /v1/models`` rows captured 2026-11-20: the Claude 5 family pins
``minimum_temperature == maximum_temperature == 1.0``; GPT-5.x/6.x lanes
publish a ``[0, 2]`` range with ``sampling_requires_reasoning_none: true``;
73 routes carry ``supports_temperature: false``.
"""

from __future__ import annotations

import importlib
import json

import pytest


def _row(mid: str, **sampling) -> dict:
    """A minimal chat/tool ``/v1/models`` row plus sampling fields."""
    row = {
        "id": mid,
        "object": "model",
        "supports_completions": True,
        "supports_tools": True,
    }
    row.update(sampling)
    return row


PINNED_ROW = _row(
    "claude-opus-5.5",
    supports_temperature=True,
    minimum_temperature=1.0,
    maximum_temperature=1.0,
    sampling_requires_reasoning_none=False,
    supports_reasoning=True,
    supported_reasoning_efforts=["low", "medium", "high"],
    reasoning_effort="medium",
)
RANGED_ROW = _row(
    "claude-haiku-4.5",
    supports_temperature=True,
    minimum_temperature=0.0,
    maximum_temperature=1.0,
    sampling_requires_reasoning_none=False,
)
REASONING_GATED_ROW = _row(
    "gpt-6-luna",
    supports_temperature=True,
    minimum_temperature=0.0,
    maximum_temperature=2.0,
    sampling_requires_reasoning_none=True,
)
NO_TEMPERATURE_ROW = _row(
    "claude-fable-latest",
    supports_temperature=False,
    minimum_temperature=None,
    maximum_temperature=None,
    sampling_requires_reasoning_none=False,
)
PLAIN_ROW = _row("deepseek-v4-flash")

SAMPLING_ROWS = [
    PINNED_ROW,
    RANGED_ROW,
    REASONING_GATED_ROW,
    NO_TEMPERATURE_ROW,
    PLAIN_ROW,
]


@pytest.fixture(scope="module")
def catalog(plugin):
    return importlib.import_module(f"{plugin.__name__}.catalog")


@pytest.fixture(scope="module")
def profile(plugin):
    # Override conftest's session fixture: providers.get_provider_profile can
    # resolve the LIVE install auto-discovered from ~/.hermes/plugins, whose
    # catalog cache is a different module instance. The worktree's own profile
    # object is the one under test (same convention as test_catalog.py).
    return plugin.experientiallabs


@pytest.fixture(scope="module")
def sampling(plugin):
    return importlib.import_module(f"{plugin.__name__}.sampling")


@pytest.fixture(autouse=True)
def _clean_catalog_cache(catalog, monkeypatch, tmp_path):
    """Isolate every test's in-memory + disk catalog cache."""
    monkeypatch.setattr(catalog, "_hermes_home_dir", lambda: tmp_path)
    catalog.invalidate_catalog_cache()
    yield
    catalog.invalidate_catalog_cache()


class TestSamplingParsing:
    def test_pinned_row_carries_all_four_fields(self, catalog):
        entry = catalog.parse_v1_model_item(PINNED_ROW)
        assert entry["supports_temperature"] is True
        assert entry["minimum_temperature"] == 1.0
        assert entry["maximum_temperature"] == 1.0
        assert entry["sampling_requires_reasoning_none"] is False

    def test_ranged_and_reasoning_gated_rows(self, catalog):
        ranged = catalog.parse_v1_model_item(RANGED_ROW)
        assert (ranged["minimum_temperature"], ranged["maximum_temperature"]) == (0.0, 1.0)
        gated = catalog.parse_v1_model_item(REASONING_GATED_ROW)
        assert gated["sampling_requires_reasoning_none"] is True
        assert gated["maximum_temperature"] == 2.0

    def test_supports_temperature_false_is_carried(self, catalog):
        entry = catalog.parse_v1_model_item(NO_TEMPERATURE_ROW)
        assert entry["supports_temperature"] is False
        # Null bounds stay absent (present-but-zero must remain distinct).
        assert "minimum_temperature" not in entry
        assert "maximum_temperature" not in entry

    def test_absent_fields_stay_absent(self, catalog):
        entry = catalog.parse_v1_model_item(PLAIN_ROW)
        for key in (
            "supports_temperature",
            "minimum_temperature",
            "maximum_temperature",
            "sampling_requires_reasoning_none",
        ):
            assert key not in entry

    @pytest.mark.parametrize("bad", [True, False, "abc", None, float("nan"), float("inf")])
    def test_malformed_bounds_are_dropped(self, catalog, bad):
        entry = catalog.parse_v1_model_item(
            _row("m", minimum_temperature=bad, maximum_temperature=bad)
        )
        assert "minimum_temperature" not in entry
        assert "maximum_temperature" not in entry

    def test_zero_and_numeric_string_bounds_parse(self, catalog):
        entry = catalog.parse_v1_model_item(
            _row("m", minimum_temperature=0, maximum_temperature="1.5")
        )
        assert entry["minimum_temperature"] == 0.0
        assert entry["maximum_temperature"] == 1.5

    @pytest.mark.parametrize("bad", ["true", 1, None])
    def test_non_bool_flags_are_dropped(self, catalog, bad):
        entry = catalog.parse_v1_model_item(
            _row("m", supports_temperature=bad, sampling_requires_reasoning_none=bad)
        )
        assert "supports_temperature" not in entry
        assert "sampling_requires_reasoning_none" not in entry

    def test_entry_stays_json_serializable_for_disk_mirror(self, catalog):
        _, entries = catalog.parse_v1_catalog(SAMPLING_ROWS)
        restored = json.loads(json.dumps(entries))
        assert restored["claude-opus-5.5"]["minimum_temperature"] == 1.0


class TestResolveTemperatureOverride:
    def test_cold_or_absent_entry_is_untouched(self, sampling):
        assert sampling.resolve_temperature_override(None) is None
        assert sampling.resolve_temperature_override({}, 0.7) is None
        assert sampling.resolve_temperature_override("not-a-dict") is None

    def test_pinned_route_forces_pinned_value(self, sampling):
        entry = {"minimum_temperature": 1.0, "maximum_temperature": 1.0}
        for caller in (None, 0.0, 0.2, 1.0, 2.0):
            assert sampling.resolve_temperature_override(entry, caller) == 1.0

    def test_range_clamps_known_caller(self, sampling):
        entry = {"minimum_temperature": 0.0, "maximum_temperature": 1.0}
        assert sampling.resolve_temperature_override(entry, 1.7) == 1.0
        assert sampling.resolve_temperature_override(entry, -0.3) == 0.0
        assert sampling.resolve_temperature_override(entry, 0.5) == 0.5

    def test_range_without_caller_value_is_untouched(self, sampling):
        entry = {"minimum_temperature": 0.0, "maximum_temperature": 2.0}
        assert sampling.resolve_temperature_override(entry) is None
        assert sampling.resolve_temperature_override(entry, None) is None

    def test_one_sided_bounds_clamp_only_when_caller_known(self, sampling):
        assert sampling.resolve_temperature_override({"minimum_temperature": 0.5}, 0.1) == 0.5
        assert sampling.resolve_temperature_override({"maximum_temperature": 1.0}, 1.5) == 1.0
        assert sampling.resolve_temperature_override({"minimum_temperature": 0.5}) is None
        assert sampling.resolve_temperature_override({"maximum_temperature": 1.0}) is None

    def test_contradictory_bounds_fail_open(self, sampling):
        entry = {"minimum_temperature": 2.0, "maximum_temperature": 1.0}
        assert sampling.resolve_temperature_override(entry, 0.5) is None

    def test_no_bounds_is_untouched(self, sampling):
        assert sampling.resolve_temperature_override({"supports_temperature": True}, 0.5) is None

    def test_supports_temperature_false_beats_bounds(self, sampling):
        entry = {
            "supports_temperature": False,
            "minimum_temperature": 1.0,
            "maximum_temperature": 1.0,
        }
        assert sampling.resolve_temperature_override(entry, 0.2) is None

    def test_reasoning_gated_sampling_needs_reasoning_off(self, sampling):
        entry = {
            "sampling_requires_reasoning_none": True,
            "minimum_temperature": 0.0,
            "maximum_temperature": 2.0,
        }
        assert sampling.resolve_temperature_override(entry, 0.5) is None
        assert (
            sampling.resolve_temperature_override(entry, 3.0, reasoning_disabled=True) == 2.0
        )
        pinned = {
            "sampling_requires_reasoning_none": True,
            "minimum_temperature": 1.0,
            "maximum_temperature": 1.0,
        }
        assert sampling.resolve_temperature_override(pinned) is None
        assert sampling.resolve_temperature_override(pinned, reasoning_disabled=True) == 1.0

    @pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), "abc", {}])
    def test_malformed_caller_value_is_untouched(self, sampling, bad):
        entry = {"minimum_temperature": 0.0, "maximum_temperature": 1.0}
        assert sampling.resolve_temperature_override(entry, bad) is None

    def test_numeric_string_caller_and_bounds_parse_defensively(self, sampling):
        entry = {"minimum_temperature": "0.0", "maximum_temperature": "1.0"}
        assert sampling.resolve_temperature_override(entry, "1.7") == 1.0

    def test_disk_mirror_entry_with_garbage_bounds_fails_open(self, sampling):
        entry = {"minimum_temperature": "hot", "maximum_temperature": None}
        assert sampling.resolve_temperature_override(entry, 0.5) is None


class TestProfileSamplingClamp:
    def test_pinned_route_gets_pinned_temperature(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        extra_body, top = profile.build_api_kwargs_extras(model="claude-opus-5.5")
        assert extra_body == {}
        assert top["temperature"] == 1.0

    def test_pinned_route_overrides_caller_temperature(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        _, top = profile.build_api_kwargs_extras(
            model="claude-opus-5.5", temperature=0.2
        )
        assert top["temperature"] == 1.0

    def test_pinned_route_with_reasoning_enabled_still_pins(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        _, top = profile.build_api_kwargs_extras(
            model="claude-opus-5.5",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert top["temperature"] == 1.0
        assert top["reasoning_effort"] == "high"

    def test_ranged_route_clamps_caller_temperature(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        _, top = profile.build_api_kwargs_extras(model="claude-haiku-4.5", temperature=1.7)
        assert top["temperature"] == 1.0
        _, top = profile.build_api_kwargs_extras(model="claude-haiku-4.5", temperature=-0.3)
        assert top["temperature"] == 0.0
        _, top = profile.build_api_kwargs_extras(model="claude-haiku-4.5", temperature=0.5)
        assert top["temperature"] == 0.5

    def test_ranged_route_without_caller_value_is_untouched(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        _, top = profile.build_api_kwargs_extras(model="claude-haiku-4.5")
        assert "temperature" not in top

    def test_cold_cache_leaves_request_untouched(self, profile, catalog):
        # No seed: cache is cold (disk mirror isolated to tmp_path).
        _, top = profile.build_api_kwargs_extras(model="claude-opus-5.5", temperature=0.2)
        assert "temperature" not in top
        _, top = profile.build_api_kwargs_extras(
            model="claude-opus-5.5", reasoning_config={"effort": "high"}
        )
        assert "temperature" not in top

    def test_supports_temperature_false_route_is_untouched(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        _, top = profile.build_api_kwargs_extras(
            model="claude-fable-latest", temperature=0.7
        )
        assert "temperature" not in top

    def test_plain_route_without_constraints_is_untouched(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        _, top = profile.build_api_kwargs_extras(model="deepseek-v4-flash", temperature=0.7)
        assert "temperature" not in top

    def test_reasoning_gated_route_omits_while_reasoning_may_be_active(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        _, top = profile.build_api_kwargs_extras(model="gpt-6-luna", temperature=0.5)
        assert "temperature" not in top
        _, top = profile.build_api_kwargs_extras(
            model="gpt-6-luna",
            reasoning_config={"enabled": True, "effort": "medium"},
            temperature=0.5,
        )
        assert "temperature" not in top

    @pytest.mark.parametrize(
        "reasoning_config", [{"enabled": False}, {"effort": "none"}]
    )
    def test_reasoning_gated_route_clamps_when_reasoning_off(
        self, profile, catalog, reasoning_config
    ):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        _, top = profile.build_api_kwargs_extras(
            model="gpt-6-luna", reasoning_config=reasoning_config, temperature=3.0
        )
        assert top["temperature"] == 2.0

    def test_suffix_and_vendor_spellings_resolve_the_pin(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        for spelling in ("claude-opus-5.5:free", "anthropic/claude-opus-5.5"):
            _, top = profile.build_api_kwargs_extras(model=spelling)
            assert top["temperature"] == 1.0

    def test_headers_and_reasoning_unchanged_alongside_clamp(self, profile, catalog):
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        extra_body, top = profile.build_api_kwargs_extras(
            model="claude-opus-5.5",
            reasoning_config={"effort": "max"},
            session_id="sess-123",
        )
        assert extra_body == {}
        assert top["temperature"] == 1.0
        # 'max' is outside the fake row's vocabulary -> clamped to 'high'.
        assert top["reasoning_effort"] == "high"
        assert top["extra_headers"]["X-Explabs-Session-Id"]
        assert "X-Explabs-Tags" in top["extra_headers"]


class TestTransportOverrideOrdering:
    """Prove the core contract the clamp relies on: top-level profile kwargs
    are applied AFTER ``_base_kwargs``, so the pin beats the caller's value."""

    def test_pinned_override_beats_caller_and_cold_cache_keeps_it(self, profile, catalog):
        from agent.transports.chat_completions import ChatCompletionsTransport

        transport = ChatCompletionsTransport()
        messages = [{"role": "user", "content": "hi"}]
        catalog.seed_catalog_from_v1_items(SAMPLING_ROWS, save_disk=False)
        kwargs = transport.build_kwargs(
            "claude-opus-5.5",
            messages,
            provider_profile=profile,
            temperature=0.2,
            base_url="https://api.experientiallabs.ai/v1",
        )
        assert kwargs["temperature"] == 1.0
        catalog.invalidate_catalog_cache()
        kwargs = transport.build_kwargs(
            "claude-opus-5.5",
            messages,
            provider_profile=profile,
            temperature=0.2,
            base_url="https://api.experientiallabs.ai/v1",
        )
        assert kwargs["temperature"] == 0.2


class TestClassifyTemperatureDecline:
    def test_unsupported_temperature_400_keeps_builtin_verdict(self, plugin):
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            error_code="unsupported_parameter",
            message="temperature is not supported for this model",
            body={"error": {"code": "unsupported_parameter", "param": "temperature"}},
            model="claude-fable-latest",
        )
        assert verdict is None

    def test_temperature_rejection_mentioning_reasoning_is_not_reasoning_mandatory(
        self, plugin
    ):
        # Wording that names BOTH temperature and a reasoning-none requirement
        # must not take the reasoning-disable recovery rung: the rejected
        # field is temperature (core's aux ladder strips it).
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            error_code="invalid_parameter",
            message="temperature requires reasoning effort none on this route",
            model="gpt-6-luna",
        )
        assert verdict is None

    def test_reasoning_mandatory_classification_still_works(self, plugin):
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            error_code="invalid_parameter",
            message="reasoning is mandatory for this model and cannot be disabled",
            model="glm-5.3-uncensored",
        )
        assert verdict == {"reason": "reasoning_mandatory", "retryable": True}
