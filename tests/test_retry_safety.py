"""W3 retry safety: 429 reset hints on classify verdicts (offline).

The classify hook runs before Hermes' built-in ``_status_429``, so its
rate-limit verdicts must carry ``error_context={"reset_at": <epoch s>}``
themselves — the shape ``max_retries_exhausted_result`` (terminal copy),
``nous_rate_guard`` and ``fallback_cooldown`` consume. These tests drive the
hook with fake OpenAI-SDK-shaped errors (``.status_code``, ``.body``,
``.response.headers``) and prove the hint extraction is defensive: missing,
malformed or hostile inputs cost the hint, never the verdict.
"""

from __future__ import annotations

import importlib
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest


class FakeResponse:
    def __init__(self, headers):
        self.headers = headers


class FakeRateLimitError(Exception):
    """Minimal OpenAI-SDK APIStatusError shape for a 429."""

    def __init__(self, *, body=None, headers=None, message="Rate limit exceeded"):
        super().__init__(message)
        self.status_code = 429
        self.body = body
        self.response = FakeResponse(headers) if headers is not None else None


class GetOnlyHeaders:
    """Headers mapping with .get() but no .items() (case-insensitive like httpx)."""

    def __init__(self, data):
        self._data = {k.lower(): v for k, v in data.items()}

    def get(self, name, default=None):
        return self._data.get(str(name).lower(), default)


class HostileHeaders:
    def items(self):
        raise RuntimeError("boom")

    def get(self, *args, **kwargs):
        raise RuntimeError("boom")


class HostileError(Exception):
    status_code = 429

    @property
    def body(self):
        raise RuntimeError("boom")

    @property
    def response(self):
        raise RuntimeError("boom")


def _rl_body(code="rate_limit_exceeded", err_type="rate_limit_error", **extra):
    return {"error": {"code": code, "type": err_type, "message": "slow down", **extra}}


def _classify(plugin, error=None, *, code="rate_limit_exceeded", body=None, message=None):
    return plugin.classify_experientiallabs_api_error(
        error, status_code=429, error_code=code, body=body, message=message
    )


def _assert_reset_near(verdict, seconds, tolerance=5.0):
    ctx = verdict.get("error_context")
    assert isinstance(ctx, dict), f"expected error_context, got {verdict!r}"
    reset_at = ctx.get("reset_at")
    assert isinstance(reset_at, float) and not isinstance(reset_at, bool)
    delay = reset_at - time.time()
    assert abs(delay - seconds) <= tolerance, f"reset in {delay:.1f}s, want ~{seconds}s"


class TestRetryAfterHeaderHints:
    @pytest.mark.parametrize(
        "code", ["rate_limit_exceeded", "admission_capacity_exceeded", "gateway_overloaded"]
    )
    def test_retry_after_seconds(self, plugin, code):
        error = FakeRateLimitError(body=_rl_body(code), headers={"Retry-After": "30"})
        verdict = _classify(plugin, error, code=code, body=error.body)
        assert verdict["reason"] == "rate_limit"
        assert verdict["retryable"] is True
        _assert_reset_near(verdict, 30)

    def test_retry_after_http_date(self, plugin):
        when = datetime.now(timezone.utc) + timedelta(seconds=120)
        error = FakeRateLimitError(headers={"Retry-After": format_datetime(when)})
        verdict = _classify(plugin, error, body=_rl_body())
        _assert_reset_near(verdict, 120)

    def test_lowercase_header_name(self, plugin):
        error = FakeRateLimitError(headers={"retry-after": "12"})
        _assert_reset_near(_classify(plugin, error), 12)

    def test_get_only_headers_mapping(self, plugin):
        error = FakeRateLimitError(headers=GetOnlyHeaders({"Retry-After": "20"}))
        _assert_reset_near(_classify(plugin, error), 20)

    def test_numeric_header_value(self, plugin):
        error = FakeRateLimitError(headers={"Retry-After": 17})
        _assert_reset_near(_classify(plugin, error), 17)

    def test_free_limit_reached_hourly_window(self, plugin):
        body = _rl_body("free_limit_reached")
        error = FakeRateLimitError(body=body, headers={"Retry-After": "45"})
        verdict = _classify(plugin, error, code="free_limit_reached", body=body)
        assert verdict["reason"] == "rate_limit"
        assert verdict["should_fallback"] is True
        _assert_reset_near(verdict, 45)

    def test_unavailable_route_overloaded_carries_hint(self, plugin):
        error = FakeRateLimitError(headers={"Retry-After": "8"})
        verdict = _classify(plugin, error, code="unavailable_route")
        assert verdict["reason"] == "overloaded"
        assert verdict["retryable"] is True
        _assert_reset_near(verdict, 8)


class TestBucketHeaderFallbacks:
    def test_x_ratelimit_reset_requests(self, plugin):
        error = FakeRateLimitError(headers={"x-ratelimit-reset-requests": "45"})
        _assert_reset_near(_classify(plugin, error), 45)

    def test_hourly_bucket_preferred_over_minute(self, plugin):
        error = FakeRateLimitError(
            headers={
                "x-ratelimit-reset-requests": "10",
                "x-ratelimit-reset-requests-1h": "3600",
            }
        )
        _assert_reset_near(_classify(plugin, error), 3600)

    def test_openai_duration_string(self, plugin):
        error = FakeRateLimitError(headers={"x-ratelimit-reset-tokens": "1m30s"})
        _assert_reset_near(_classify(plugin, error), 90)

    def test_x_ratelimit_reset_epoch(self, plugin):
        target = time.time() + 500
        error = FakeRateLimitError(headers={"x-ratelimit-reset": str(int(target))})
        _assert_reset_near(_classify(plugin, error), 500)

    def test_retry_after_wins_over_bucket_headers(self, plugin):
        error = FakeRateLimitError(
            headers={"Retry-After": "30", "x-ratelimit-reset-requests-1h": "3600"}
        )
        _assert_reset_near(_classify(plugin, error), 30)


class TestBodyResetSources:
    def test_body_retry_after(self, plugin):
        body = _rl_body(retry_after=12.5)
        verdict = _classify(plugin, FakeRateLimitError(body=body), body=body)
        _assert_reset_near(verdict, 12.5)

    def test_body_resets_at_epoch(self, plugin):
        target = time.time() + 200
        body = _rl_body(resets_at=int(target))
        _assert_reset_near(_classify(plugin, body=body), 200)

    def test_body_resets_at_epoch_ms(self, plugin):
        target = time.time() + 200
        body = _rl_body(resets_at=int(target * 1000))
        _assert_reset_near(_classify(plugin, body=body), 200)

    def test_body_resets_at_iso(self, plugin):
        when = datetime.now(timezone.utc) + timedelta(seconds=300)
        body = _rl_body(resets_at=when.isoformat())
        _assert_reset_near(_classify(plugin, body=body), 300)

    def test_body_resets_in_seconds(self, plugin):
        body = _rl_body(resets_in_seconds=90)
        _assert_reset_near(_classify(plugin, body=body), 90)

    def test_top_level_body_fields(self, plugin):
        body = {"code": "rate_limit_exceeded", "retry_after": 20}
        _assert_reset_near(_classify(plugin, body=body), 20)

    def test_absolute_body_reset_wins_over_header(self, plugin):
        target = time.time() + 500
        body = _rl_body(resets_at=int(target))
        error = FakeRateLimitError(body=body, headers={"Retry-After": "30"})
        _assert_reset_near(_classify(plugin, error, body=body), 500)

    def test_error_body_used_when_body_kwarg_missing(self, plugin):
        error = FakeRateLimitError(body=_rl_body(retry_after=25))
        _assert_reset_near(_classify(plugin, error), 25)


class TestMessageGrammar:
    def test_retry_after_text(self, plugin):
        verdict = _classify(
            plugin, message="rate limit exceeded for org; retry after 45 seconds"
        )
        _assert_reset_near(verdict, 45)

    def test_resets_in_text(self, plugin):
        verdict = _classify(plugin, message="free limit reached; resets in 1hr 5min")
        _assert_reset_near(verdict, 3900)

    def test_error_str_fallback(self, plugin):
        error = FakeRateLimitError(message="Error code: 429 - retry after 30 s")
        _assert_reset_near(_classify(plugin, error), 30)

    def test_body_message_fallback(self, plugin):
        body = {
            "error": {
                "code": "free_limit_reached",
                "type": "rate_limit_error",
                "message": "Hourly free limit reached; resets in 30m.",
            }
        }
        _assert_reset_near(_classify(plugin, body=body), 1800)


class TestNoHintKeepsVerdict:
    @pytest.mark.parametrize(
        "value",
        ["soon", "", "   ", None, True, "-5", "0", 0, "nan", "inf", "1e30", object()],
    )
    def test_malformed_retry_after(self, plugin, value):
        error = FakeRateLimitError(headers={"Retry-After": value})
        verdict = _classify(plugin, error)
        assert verdict == {"reason": "rate_limit", "retryable": True}

    def test_none_error(self, plugin):
        assert _classify(plugin, None) == {"reason": "rate_limit", "retryable": True}

    def test_error_without_response(self, plugin):
        error = Exception("429 slow down")
        error.status_code = 429
        error.body = _rl_body()
        assert _classify(plugin, error) == {"reason": "rate_limit", "retryable": True}

    def test_response_without_headers(self, plugin):
        error = FakeRateLimitError()
        error.response = FakeResponse(None)
        assert _classify(plugin, error) == {"reason": "rate_limit", "retryable": True}

    def test_hostile_headers(self, plugin):
        error = FakeRateLimitError(headers=HostileHeaders())
        assert _classify(plugin, error) == {"reason": "rate_limit", "retryable": True}

    def test_hostile_error_properties(self, plugin):
        verdict = _classify(plugin, HostileError("429"))
        assert verdict == {"reason": "rate_limit", "retryable": True}

    @pytest.mark.parametrize("body", [["not", "a", "dict"], "text", object(), 42])
    def test_non_dict_body(self, plugin, body):
        verdict = _classify(plugin, body=body)
        assert verdict == {"reason": "rate_limit", "retryable": True}

    def test_expired_reset_is_no_hint(self, plugin):
        body = _rl_body(resets_at=int(time.time()) - 100)
        assert _classify(plugin, body=body) == {"reason": "rate_limit", "retryable": True}

    def test_absurd_future_reset_is_no_hint(self, plugin):
        body = _rl_body(resets_at=int(time.time()) + 10**9)
        assert _classify(plugin, body=body) == {"reason": "rate_limit", "retryable": True}

    @pytest.mark.parametrize(
        "code,err_type,expected",
        [
            ("insufficient_quota", "insufficient_quota", "billing"),
            ("promo_credits_now", "invalid_request_error", "billing"),
            ("free_limit_reached", "insufficient_quota", "billing"),
        ],
    )
    def test_billing_verdicts_stay_hint_free(self, plugin, code, err_type, expected):
        body = _rl_body(code, err_type)
        error = FakeRateLimitError(body=body, headers={"Retry-After": "30"})
        verdict = _classify(plugin, error, code=code, body=body)
        assert verdict["reason"] == expected
        assert verdict["retryable"] is False
        assert "error_context" not in verdict

    def test_unknown_429_code_keeps_builtin_handling(self, plugin):
        error = FakeRateLimitError(headers={"Retry-After": "30"})
        assert _classify(plugin, error, code="mystery_code") is None

    def test_non_429_verdicts_unaffected(self, plugin):
        error = FakeRateLimitError(headers={"Retry-After": "30"})
        error.status_code = 400
        verdict = plugin.classify_experientiallabs_api_error(
            error,
            status_code=400,
            error_code="content_filter",
            message="a gateway guardrail could not complete this request.",
        )
        assert verdict == {"reason": "content_policy_blocked", "retryable": True}


class TestHermesWiring:
    """End-to-end through the real classifier: verdict -> ClassifiedError."""

    def _classified(self, plugin, monkeypatch, error):
        import providers

        monkeypatch.setattr(
            providers,
            "get_provider_profile",
            lambda name: plugin.experientiallabs if name == "experientiallabs" else None,
        )
        from agent.error_classifier import classify_api_error

        return classify_api_error(error, provider="experientiallabs", model="gpt-6-luna")

    def test_classified_error_carries_reset_at(self, plugin, monkeypatch):
        error = FakeRateLimitError(body=_rl_body(), headers={"Retry-After": "30"})
        classified = self._classified(plugin, monkeypatch, error)
        assert classified.reason.value == "rate_limit"
        assert classified.retryable is True
        reset_at = classified.error_context.get("reset_at")
        assert isinstance(reset_at, float)
        assert 25 <= reset_at - time.time() <= 35

    def test_reset_shape_matches_fallback_cooldown_consumer(self, plugin, monkeypatch):
        # fallback_cooldown._provider_reset_delay is the pure parser the
        # fallback walk applies to reset_at (epoch s, future); prove the hint
        # round-trips through it instead of landing as "expired/invalid".
        from agent.fallback_cooldown import _provider_reset_delay

        error = FakeRateLimitError(headers={"Retry-After": "42"})
        classified = self._classified(plugin, monkeypatch, error)
        delay = _provider_reset_delay(classified.error_context.get("reset_at"))
        assert delay is not None
        assert 37 <= delay <= 47

    def test_no_headers_keeps_builtin_context_path(self, plugin, monkeypatch):
        # Without a plugin hint the verdict still classifies; the built-in
        # extractor reads the raw error independently (no error_context here).
        classified = self._classified(plugin, monkeypatch, FakeRateLimitError())
        assert classified.reason.value == "rate_limit"
        assert classified.error_context.get("reset_at") is None


class TestRetryHintsModule:
    @pytest.fixture()
    def hints(self, plugin):
        return importlib.import_module(f"{plugin.__name__}.retry_hints")

    def test_context_empty_without_signal(self, hints):
        assert hints.rate_limit_error_context(None) == {}
        assert hints.rate_limit_error_context(FakeRateLimitError()) == {}

    def test_context_shape(self, hints):
        error = FakeRateLimitError(headers={"Retry-After": "60"})
        ctx = hints.rate_limit_error_context(error)
        assert set(ctx) == {"reset_at"}
        assert 55 <= ctx["reset_at"] - time.time() <= 65

    def test_reset_epoch_never_raises_on_junk(self, hints):
        for junk in (None, object(), 42, "text", [], {"error": None}):
            assert hints.reset_epoch_from_error(junk) is None
        assert hints.reset_epoch_from_error(HostileError("x")) is None

    def test_body_kwarg_wins_over_error_body(self, hints):
        error = FakeRateLimitError(body=_rl_body(retry_after=10))
        epoch = hints.reset_epoch_from_error(error, body=_rl_body(retry_after=80))
        assert 75 <= epoch - time.time() <= 85
