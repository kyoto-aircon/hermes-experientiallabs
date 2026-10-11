"""Tests for the Experiential Labs account-usage builders and fetch hook.

Builders are pure and tested directly; the fetch path is exercised with the
network boundary stubbed so no live call is made.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal

import pytest


def _window_by_label(snapshot, label):
    for window in snapshot.windows:
        if window.label == label:
            return window
    return None


class TestHelpers:
    def test_to_decimal_accepts_numeric_shapes(self, account_usage):
        assert account_usage._to_decimal(1) == Decimal("1")
        assert account_usage._to_decimal("2.5") == Decimal("2.5")
        assert account_usage._to_decimal(" 3 ") == Decimal("3")

    @pytest.mark.parametrize("value", [None, "", "abc", [], {}, True, False, "NaN", "Infinity"])
    def test_to_decimal_rejects_non_finite_and_non_numeric(self, account_usage, value):
        assert account_usage._to_decimal(value) is None

    def test_unwrap_data_object(self, account_usage):
        assert account_usage._unwrap({"data": {"a": 1}}) == {"a": 1}

    def test_unwrap_bare_top_level(self, account_usage):
        assert account_usage._unwrap({"a": 1}) == {"a": 1}

    def test_unwrap_non_dict_is_empty(self, account_usage):
        assert account_usage._unwrap([1, 2]) == {}
        assert account_usage._unwrap(None) == {}

    def test_pick_keeps_present_zero(self, account_usage):
        assert account_usage._pick({"limit": 0}, "limit") == 0

    def test_pick_skips_none_and_missing(self, account_usage):
        assert account_usage._pick({"limit": None}, "limit") is account_usage._MISSING
        assert account_usage._pick({}, "limit") is account_usage._MISSING

    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            Decimal("NaN"),
            Decimal("Infinity"),
            Decimal("-Infinity"),
            "nan",
            "inf",
            "-inf",
            "   ",
        ],
    )
    def test_to_decimal_rejects_non_finite_variants(self, account_usage, value):
        # Non-finite numbers can arrive as real floats/Decimals or as their
        # string spellings; all must fail closed to None.
        assert account_usage._to_decimal(value) is None

    def test_to_decimal_accepts_decimal_and_negative(self, account_usage):
        assert account_usage._to_decimal(Decimal("4.25")) == Decimal("4.25")
        assert account_usage._to_decimal("-7") == Decimal("-7")
        assert account_usage._to_decimal(0) == Decimal("0")

    def test_pick_present_zero_beats_later_key(self, account_usage):
        # A configured $0 must not silently fall through to a later alias.
        assert account_usage._pick({"limit": 0, "cap": 5}, "limit", "cap") == 0

    def test_pick_falls_through_none_to_later_key(self, account_usage):
        assert account_usage._pick({"limit": None, "cap": 5}, "limit", "cap") == 5

    def test_unwrap_data_non_dict_returns_bare_payload(self, account_usage):
        # ``data`` present but not an object: fall back to the top-level body.
        assert account_usage._unwrap({"data": None}) == {"data": None}
        assert account_usage._unwrap({"data": [1, 2]}) == {"data": [1, 2]}


class TestNextUtcMidnight:
    def test_rolls_to_next_day(self, account_usage):
        now = datetime(2026, 10, 6, 13, 45, tzinfo=timezone.utc)
        assert account_usage.next_utc_midnight(now) == datetime(
            2026, 10, 7, 0, 0, tzinfo=timezone.utc
        )

    def test_handles_month_boundary(self, account_usage):
        now = datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc)
        assert account_usage.next_utc_midnight(now) == datetime(
            2026, 11, 1, 0, 0, tzinfo=timezone.utc
        )

    def test_naive_datetime_treated_as_utc(self, account_usage):
        now = datetime(2026, 10, 6, 1, 0)
        assert account_usage.next_utc_midnight(now) == datetime(
            2026, 10, 7, 0, 0, tzinfo=timezone.utc
        )

    def test_defaults_to_now(self, account_usage):
        reset = account_usage.next_utc_midnight()
        assert reset.tzinfo is timezone.utc
        assert (reset.hour, reset.minute, reset.second) == (0, 0, 0)


class TestWalletWindow:
    def test_computes_remaining_and_percent(self, account_usage):
        window = account_usage.build_wallet_window(
            {"data": {"total_credits": "10", "total_usage": "2.5"}}
        )
        assert window.label == "Credits"
        assert window.used_percent == pytest.approx(25.0)
        assert "7.50" in window.detail
        assert "2.50" in window.detail

    def test_zero_total_does_not_divide_by_zero(self, account_usage):
        window = account_usage.build_wallet_window(
            {"data": {"total_credits": "0", "total_usage": "0"}}
        )
        assert window.used_percent == 0.0
        assert "0.00" in window.detail

    def test_overdrawn_clamps_to_full_percent(self, account_usage):
        window = account_usage.build_wallet_window(
            {"data": {"total_credits": "1", "total_usage": "3"}}
        )
        assert window.used_percent == 100.0
        assert "0.00" in window.detail

    def test_accepts_bare_top_level_body(self, account_usage):
        window = account_usage.build_wallet_window(
            {"total_credits": "5", "total_usage": "1"}
        )
        assert window is not None
        assert window.used_percent == pytest.approx(20.0)

    def test_negative_usage_clamps_percent_to_zero(self, account_usage):
        window = account_usage.build_wallet_window(
            {"data": {"total_credits": "10", "total_usage": "-5"}}
        )
        assert window is not None
        assert window.used_percent == 0.0

    def test_nonpositive_total_yields_zero_percent_and_zero_remaining(self, account_usage):
        window = account_usage.build_wallet_window(
            {"data": {"total_credits": "-5", "total_usage": "1"}}
        )
        assert window is not None
        assert window.used_percent == 0.0
        assert "0.00" in window.detail

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"data": {}},
            {"data": {"total_credits": "10"}},
            {"data": {"total_usage": "1"}},
            {"data": {"total_credits": "abc", "total_usage": "1"}},
            {"data": "not-a-dict"},
            None,
        ],
    )
    def test_missing_or_invalid_fields_return_none(self, account_usage, payload):
        assert account_usage.build_wallet_window(payload) is None


class TestDailyCapWindow:
    def test_computes_used_percent_and_reset(self, account_usage):
        now = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
        window = account_usage.build_daily_cap_window(
            {"data": {"limit": "5", "limit_remaining": "2"}}, now=now
        )
        assert window.label == "Daily key cap"
        assert window.used_percent == pytest.approx(60.0)
        assert window.reset_at == datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc)
        assert "3.00" in window.detail and "5.00" in window.detail

    def test_no_cap_returns_none(self, account_usage):
        assert account_usage.build_daily_cap_window({"data": {"limit": None}}) is None
        assert account_usage.build_daily_cap_window({"data": {}}) is None

    def test_hard_zero_cap_is_flagged(self, account_usage):
        window = account_usage.build_daily_cap_window(
            {"data": {"limit": 0, "limit_remaining": 0}}
        )
        assert window.used_percent == 100.0
        assert "hard" in window.detail

    def test_missing_remaining_returns_none(self, account_usage):
        assert account_usage.build_daily_cap_window({"data": {"limit": "5"}}) is None

    def test_non_numeric_returns_none(self, account_usage):
        assert account_usage.build_daily_cap_window(
            {"data": {"limit": "x", "limit_remaining": "1"}}
        ) is None
        assert account_usage.build_daily_cap_window(
            {"data": {"limit": "5", "limit_remaining": "y"}}
        ) is None

    def test_negative_limit_is_treated_as_cap(self, account_usage):
        window = account_usage.build_daily_cap_window(
            {"data": {"limit": "-1", "limit_remaining": "0"}}
        )
        assert window.used_percent == 100.0

    def test_alternate_field_names(self, account_usage):
        window = account_usage.build_daily_cap_window(
            {"data": {"daily_limit": "4", "remaining": "1"}}
        )
        assert window is not None
        assert window.used_percent == pytest.approx(75.0)

    def test_cap_and_limitRemaining_alternate_names(self, account_usage):
        window = account_usage.build_daily_cap_window(
            {"data": {"cap": "2", "limitRemaining": "0.5"}}
        )
        assert window is not None
        assert window.used_percent == pytest.approx(75.0)

    def test_positive_limit_with_present_zero_remaining_is_full(self, account_usage):
        # Present-zero remaining is distinct from "field absent": cap is spent.
        window = account_usage.build_daily_cap_window(
            {"data": {"limit": "5", "limit_remaining": "0"}}
        )
        assert window is not None
        assert window.used_percent == 100.0
        assert "hard" not in window.detail

    def test_remaining_exceeding_limit_clamps_to_zero_percent(self, account_usage):
        window = account_usage.build_daily_cap_window(
            {"data": {"limit": "5", "limit_remaining": "8"}}
        )
        assert window is not None
        assert window.used_percent == 0.0

    def test_hard_zero_cap_sets_utc_midnight_reset(self, account_usage):
        now = datetime(2026, 10, 6, 23, 59, tzinfo=timezone.utc)
        window = account_usage.build_daily_cap_window(
            {"data": {"limit": 0, "limit_remaining": 0}}, now=now
        )
        assert window is not None
        assert window.used_percent == 100.0
        assert window.reset_at == datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc)

    def test_zero_alternate_limit_is_hard_cap(self, account_usage):
        window = account_usage.build_daily_cap_window(
            {"data": {"daily_limit": 0, "remaining": 0}}
        )
        assert window is not None
        assert window.used_percent == 100.0

    def test_hard_zero_cap_without_remaining_is_still_flagged(self, account_usage):
        # A disabled/hard cap may omit limit_remaining; the $0 cap must survive.
        window = account_usage.build_daily_cap_window({"data": {"limit": 0}})
        assert window is not None
        assert window.used_percent == 100.0
        assert "hard" in window.detail

    def test_limit_without_any_remaining_alias_returns_none(self, account_usage):
        assert account_usage.build_daily_cap_window({"data": {"cap": "5"}}) is None
        assert account_usage.build_daily_cap_window(
            {"data": {"cap": "5", "limitRemaining": None}}
        ) is None


class _FakeResponse:
    def __init__(self, *, status_code=200, payload=None, raises_on_json=False):
        self.status_code = status_code
        self._payload = payload
        self._raises_on_json = raises_on_json

    def json(self):
        if self._raises_on_json:
            raise ValueError("not json")
        return self._payload


class _FakeClient:
    def __init__(self, *, response=None, raises_on_get=None):
        self._response = response
        self._raises_on_get = raises_on_get

    def get(self, _url, headers=None):  # noqa: ARG002 - signature matches httpx
        if self._raises_on_get is not None:
            raise self._raises_on_get
        return self._response


class TestGetJson:
    def test_returns_decoded_json_on_200(self, account_usage):
        client = _FakeClient(response=_FakeResponse(payload={"data": {"a": 1}}))
        assert account_usage._get_json(client, "https://x", {}) == {"data": {"a": 1}}

    def test_non_200_returns_none(self, account_usage):
        client = _FakeClient(response=_FakeResponse(status_code=500, payload={"a": 1}))
        assert account_usage._get_json(client, "https://x", {}) is None

    def test_transport_exception_returns_none(self, account_usage):
        client = _FakeClient(raises_on_get=RuntimeError("boom"))
        assert account_usage._get_json(client, "https://x", {}) is None

    def test_json_decode_exception_returns_none(self, account_usage):
        client = _FakeClient(response=_FakeResponse(raises_on_json=True))
        assert account_usage._get_json(client, "https://x", {}) is None


class TestUsageSettings:
    def test_cost_urls_default_on_missing_empty_or_non_string(self, account_usage):
        defaults = (account_usage._CREDITS_URL, account_usage._KEY_URL)
        assert account_usage._cost_urls(None) == defaults
        assert account_usage._cost_urls("") == defaults
        assert account_usage._cost_urls("   ") == defaults
        assert account_usage._cost_urls(5) == defaults
        # Slash-only input must not collapse into relative URLs.
        assert account_usage._cost_urls("///") == defaults

    def test_cost_urls_join_paths_and_strip_trailing_slash(self, account_usage):
        assert account_usage._cost_urls("https://mirror.test/api/v1") == (
            "https://mirror.test/api/v1/credits",
            "https://mirror.test/api/v1/key",
        )
        assert account_usage._cost_urls("https://mirror.test/api/v1/") == (
            "https://mirror.test/api/v1/credits",
            "https://mirror.test/api/v1/key",
        )
        assert account_usage._cost_urls("https://mirror.test/api/v1///") == (
            "https://mirror.test/api/v1/credits",
            "https://mirror.test/api/v1/key",
        )
        assert account_usage._cost_urls("  https://mirror.test/api/v1/  ") == (
            "https://mirror.test/api/v1/credits",
            "https://mirror.test/api/v1/key",
        )

    @pytest.mark.parametrize(
        "value",
        [None, "abc", "", True, False, 0, -2, float("nan"), float("inf"), float("-inf"), [], {}],
    )
    def test_usage_timeout_defaults_on_invalid_values(self, account_usage, value):
        assert account_usage._usage_timeout(value) == account_usage._HTTP_TIMEOUT_S

    def test_usage_timeout_accepts_numbers_and_numeric_strings(self, account_usage):
        assert account_usage._usage_timeout(2) == 2.0
        assert account_usage._usage_timeout(2.5) == 2.5
        assert account_usage._usage_timeout("2.5") == 2.5

    def test_usage_timeout_clamped_under_hook_deadline(self, account_usage):
        # Two sequential GETs at the clamp must fit Hermes' 10s usage-hook
        # deadline with room to spare (2 x 4.5s = 9s).
        assert account_usage._MAX_TIMEOUT_S <= 4.5
        assert account_usage._usage_timeout(30) == account_usage._MAX_TIMEOUT_S
        assert account_usage._usage_timeout("5.5") == account_usage._MAX_TIMEOUT_S
        assert account_usage._usage_timeout(5.0) == account_usage._MAX_TIMEOUT_S
        # The clamp is min-semantics: the boundary itself is allowed (the
        # manifest documents "values above 4.5 are clamped"), never pushed
        # below.
        assert account_usage._usage_timeout(4.5) == 4.5
        assert account_usage._usage_timeout(4.4) == 4.4


class TestResolveUsageSettings:
    def _patch_config(self, monkeypatch, config):
        import hermes_cli.config as config_mod

        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: config)

    def _defaults(self, account_usage):
        return (
            account_usage._CREDITS_URL,
            account_usage._KEY_URL,
            account_usage._HTTP_TIMEOUT_S,
        )

    def test_reads_overrides_from_plugin_settings(self, account_usage, monkeypatch):
        self._patch_config(
            monkeypatch,
            {
                "plugins": {
                    "entries": {
                        "hermes-experientiallabs": {
                            "settings": {
                                "cost_api_base": "https://mirror.test/api/v1",
                                "usage_timeout_s": 2.0,
                            }
                        }
                    }
                }
            },
        )
        assert account_usage.resolve_usage_settings() == (
            "https://mirror.test/api/v1/credits",
            "https://mirror.test/api/v1/key",
            2.0,
        )

    def test_defaults_without_plugin_entry(self, account_usage, monkeypatch):
        self._patch_config(monkeypatch, {"plugins": {"entries": {}}})
        assert account_usage.resolve_usage_settings() == self._defaults(account_usage)

    def test_defaults_without_settings_block(self, account_usage, monkeypatch):
        self._patch_config(
            monkeypatch, {"plugins": {"entries": {"hermes-experientiallabs": {}}}}
        )
        assert account_usage.resolve_usage_settings() == self._defaults(account_usage)

    def test_defaults_when_settings_not_a_mapping(self, account_usage, monkeypatch):
        self._patch_config(
            monkeypatch,
            {"plugins": {"entries": {"hermes-experientiallabs": {"settings": "junk"}}}},
        )
        assert account_usage.resolve_usage_settings() == self._defaults(account_usage)

    def test_defaults_on_unexpected_config_shape(self, account_usage, monkeypatch):
        self._patch_config(monkeypatch, {"plugins": "not-a-mapping"})
        assert account_usage.resolve_usage_settings() == self._defaults(account_usage)

    def test_defaults_when_loader_raises(self, account_usage, monkeypatch):
        import hermes_cli.config as config_mod

        def boom():
            raise RuntimeError("config still initializing")

        monkeypatch.setattr(config_mod, "load_config_readonly", boom)
        assert account_usage.resolve_usage_settings() == self._defaults(account_usage)

    def test_defaults_when_config_import_fails(self, account_usage, monkeypatch):
        # The lazy ``hermes_cli.config`` import itself failing (broken or
        # half-initialized install) must fail open like every other error.
        monkeypatch.setitem(sys.modules, "hermes_cli.config", None)
        with pytest.raises(ImportError):
            from hermes_cli.config import load_config_readonly  # noqa: F401
        assert account_usage.resolve_usage_settings() == self._defaults(account_usage)

    def test_partial_overrides_keep_other_defaults(self, account_usage, monkeypatch):
        self._patch_config(
            monkeypatch,
            {
                "plugins": {
                    "entries": {
                        "hermes-experientiallabs": {
                            "settings": {"usage_timeout_s": 1.5}
                        }
                    }
                }
            },
        )
        assert account_usage.resolve_usage_settings() == (
            account_usage._CREDITS_URL,
            account_usage._KEY_URL,
            1.5,
        )

    def test_invalid_setting_values_fall_back_per_key(self, account_usage, monkeypatch):
        # A hand-edited config.yaml can carry junk for both keys; each key
        # must fall back to its default independently, never raise.
        self._patch_config(
            monkeypatch,
            {
                "plugins": {
                    "entries": {
                        "hermes-experientiallabs": {
                            "settings": {"cost_api_base": 42, "usage_timeout_s": "soon"}
                        }
                    }
                }
            },
        )
        assert account_usage.resolve_usage_settings() == self._defaults(account_usage)

    def test_invalid_base_with_valid_timeout_keeps_timeout(self, account_usage, monkeypatch):
        # Mixed junk: the bad base falls back while the good timeout survives.
        self._patch_config(
            monkeypatch,
            {
                "plugins": {
                    "entries": {
                        "hermes-experientiallabs": {
                            "settings": {"cost_api_base": [], "usage_timeout_s": 3.0}
                        }
                    }
                }
            },
        )
        assert account_usage.resolve_usage_settings() == (
            account_usage._CREDITS_URL,
            account_usage._KEY_URL,
            3.0,
        )


class TestFetchHook:
    def _patch_runtime(self, monkeypatch, token):
        import hermes_cli.runtime_provider as rp

        monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **_: {"api_key": token})

    def test_no_token_returns_none(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "")
        assert account_usage.fetch_experientiallabs_account_usage("experientiallabs") is None

    def test_both_endpoints_fail_returns_none(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "xpl_key")
        monkeypatch.setattr(account_usage, "_get_json", lambda *a, **k: None)
        assert account_usage.fetch_experientiallabs_account_usage("experientiallabs") is None

    def test_returns_snapshot_with_both_windows(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "xpl_key")

        def fake_get_json(_client, url, _headers):
            if url.endswith("/credits"):
                return {"data": {"total_credits": "10", "total_usage": "2"}}
            if url.endswith("/key"):
                return {"data": {"limit": "5", "limit_remaining": "3"}}
            return None

        monkeypatch.setattr(account_usage, "_get_json", fake_get_json)
        snapshot = account_usage.fetch_experientiallabs_account_usage("experientiallabs")

        assert snapshot is not None
        assert snapshot.provider == "experientiallabs"
        assert snapshot.source == "explabs_cost_api"
        assert snapshot.available is True
        assert _window_by_label(snapshot, "Credits") is not None
        assert _window_by_label(snapshot, "Daily key cap") is not None

    def test_one_endpoint_failing_still_yields_snapshot(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "xpl_key")

        def fake_get_json(_client, url, _headers):
            if url.endswith("/credits"):
                return {"data": {"total_credits": "10", "total_usage": "2"}}
            return None

        monkeypatch.setattr(account_usage, "_get_json", fake_get_json)
        snapshot = account_usage.fetch_experientiallabs_account_usage("experientiallabs")

        assert snapshot is not None
        assert _window_by_label(snapshot, "Credits") is not None
        assert _window_by_label(snapshot, "Daily key cap") is None

    def test_unparseable_bodies_return_none(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "xpl_key")
        monkeypatch.setattr(account_usage, "_get_json", lambda *a, **k: {"data": {}})
        assert account_usage.fetch_experientiallabs_account_usage("experientiallabs") is None

    def test_whitespace_token_returns_none(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "   ")
        assert account_usage.fetch_experientiallabs_account_usage("experientiallabs") is None

    def test_missing_api_key_returns_none(self, account_usage, monkeypatch):
        import hermes_cli.runtime_provider as rp

        monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **_: {})
        assert account_usage.fetch_experientiallabs_account_usage("experientiallabs") is None

    def test_resolver_raise_returns_none(self, account_usage, monkeypatch):
        # The hook must never raise, even when credential resolution fails
        # (e.g. no key configured / provider disabled).
        import hermes_cli.runtime_provider as rp

        def boom(**_):
            raise RuntimeError("no credential")

        monkeypatch.setattr(rp, "resolve_runtime_provider", boom)
        assert account_usage.fetch_experientiallabs_account_usage("experientiallabs") is None

    def test_forwards_explicit_base_url_and_api_key(self, account_usage, monkeypatch):
        import hermes_cli.runtime_provider as rp

        captured = {}

        def fake_resolve(**kwargs):
            captured.update(kwargs)
            return {"api_key": "xpl_key"}

        monkeypatch.setattr(rp, "resolve_runtime_provider", fake_resolve)
        monkeypatch.setattr(account_usage, "_get_json", lambda *a, **k: None)
        account_usage.fetch_experientiallabs_account_usage(
            "experientiallabs", base_url="https://example.test/v1", api_key="tok"
        )
        assert captured == {
            "requested": "experientiallabs",
            "explicit_base_url": "https://example.test/v1",
            "explicit_api_key": "tok",
        }

    def test_only_key_endpoint_succeeds(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "xpl_key")

        def fake_get_json(_client, url, _headers):
            if url.endswith("/key"):
                return {"data": {"limit": "5", "limit_remaining": "2"}}
            return None

        monkeypatch.setattr(account_usage, "_get_json", fake_get_json)
        snapshot = account_usage.fetch_experientiallabs_account_usage("experientiallabs")

        assert snapshot is not None
        assert _window_by_label(snapshot, "Credits") is None
        assert _window_by_label(snapshot, "Daily key cap") is not None

    def test_unparseable_success_with_other_failure_returns_none(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "xpl_key")

        def fake_get_json(_client, url, _headers):
            if url.endswith("/credits"):
                # 200 but missing the required fields -> builder yields None.
                return {"data": {"unexpected": "shape"}}
            return None

        monkeypatch.setattr(account_usage, "_get_json", fake_get_json)
        assert account_usage.fetch_experientiallabs_account_usage("experientiallabs") is None

    def test_settings_reach_the_fetch_urls_and_timeout(self, account_usage, monkeypatch):
        self._patch_runtime(monkeypatch, "xpl_key")

        def fake_config():
            return {
                "plugins": {
                    "entries": {
                        "hermes-experientiallabs": {
                            "settings": {
                                "cost_api_base": "https://mirror.test/api/v1",
                                "usage_timeout_s": 2.5,
                            }
                        }
                    }
                }
            }

        import hermes_cli.config as config_mod
        import httpx

        monkeypatch.setattr(config_mod, "load_config_readonly", fake_config)

        seen = {"urls": [], "timeout": None}

        class _RecordingClient:
            def __init__(self, timeout=None, follow_redirects=False):
                seen["timeout"] = timeout

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url, headers=None):
                seen["urls"].append(url)
                return _FakeResponse(payload={"data": {"total_credits": "1", "total_usage": "0"}})

        monkeypatch.setattr(httpx, "Client", _RecordingClient)
        snapshot = account_usage.fetch_experientiallabs_account_usage("experientiallabs")

        assert seen["timeout"] == 2.5
        assert seen["urls"] == [
            "https://mirror.test/api/v1/credits",
            "https://mirror.test/api/v1/key",
        ]
        assert snapshot is not None
        assert _window_by_label(snapshot, "Credits") is not None

    def test_default_settings_reach_the_fetch_urls_and_timeout(self, account_usage, monkeypatch):
        # Empty plugin settings: the fetch must use the built-in default URLs
        # and timeout — pinned here with the config loader stubbed, so it does
        # not depend on whatever the machine's real config.yaml happens to say.
        self._patch_runtime(monkeypatch, "xpl_key")

        import hermes_cli.config as config_mod
        import httpx

        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {})

        seen = {"urls": [], "timeout": None}

        class _RecordingClient:
            def __init__(self, timeout=None, follow_redirects=False):
                seen["timeout"] = timeout

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url, headers=None):
                seen["urls"].append(url)
                return _FakeResponse(payload={"data": {"total_credits": "1", "total_usage": "0"}})

        monkeypatch.setattr(httpx, "Client", _RecordingClient)
        snapshot = account_usage.fetch_experientiallabs_account_usage("experientiallabs")

        assert seen["timeout"] == account_usage._HTTP_TIMEOUT_S
        assert seen["urls"] == [account_usage._CREDITS_URL, account_usage._KEY_URL]
        assert snapshot is not None
        assert _window_by_label(snapshot, "Credits") is not None
