"""Unit tests for the gateway-guardrail error classification hook.

The hosted gateway's content-inspection service intermittently answers HTTP 400
before a request reaches a model. Hermes' built-in classifier reads a
``content_filter`` 400 as a deterministic content-policy block and ends the
turn; this hook marks exactly the two observed gateway-service signatures
retryable so Hermes backs off and retries. Real content refusals (different
messages) keep the built-in verdict.
"""

from __future__ import annotations

import json

import pytest

INSPECTION_MESSAGE = (
    "Content inspection does not support this complete request. "
    "Use a supported context or modality."
)
GUARDRAIL_MESSAGE = "A gateway guardrail could not complete this request."


class FakeBadRequest(Exception):
    """Minimal OpenAI-SDK-shaped 400: status on the exception, body dict."""

    def __init__(self, body):
        super().__init__(json.dumps(body))
        self.status_code = 400
        self.body = body


def _body(code: str, message: str) -> dict:
    return {"error": {"message": message, "code": code}}


class TestGatewayGuardrailSignatures:
    @pytest.mark.parametrize(
        "code", ["unsupported_capability", "content_filter"]
    )
    @pytest.mark.parametrize(
        "message", [INSPECTION_MESSAGE, GUARDRAIL_MESSAGE]
    )
    def test_observed_400s_are_retryable(self, plugin, code, message):
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            error_code=code,
            # Hermes hands the hook a lowercased message (_Ctx.msg).
            message=message.lower(),
            model="glm-5.3-uncensored",
        )
        assert verdict == {"reason": "content_policy_blocked", "retryable": True}

    def test_message_match_is_case_insensitive(self, plugin):
        # The classifier lowercases, but a direct/other caller may not.
        verdict = plugin.classify_experientiallabs_api_error(
            None, status_code=400, error_code="content_filter", message=GUARDRAIL_MESSAGE
        )
        assert verdict["retryable"] is True

    def test_truncated_message_still_matches(self, plugin):
        # The markers deliberately omit the trailing period so a gateway that
        # truncates or re-punctuates the sentence still matches.
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            error_code="content_filter",
            message="a gateway guardrail could not complete this request",
        )
        assert verdict == {"reason": "content_policy_blocked", "retryable": True}

    def test_code_and_message_fall_back_to_the_response_body(self, plugin):
        # Some SDK paths expose only the decoded body.
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            body=_body("unsupported_capability", INSPECTION_MESSAGE),
        )
        assert verdict == {"reason": "content_policy_blocked", "retryable": True}

    def test_explicit_code_wins_over_the_body(self, plugin):
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            error_code="invalid_request_error",
            message=GUARDRAIL_MESSAGE.lower(),
            body=_body("content_filter", GUARDRAIL_MESSAGE),
        )
        assert verdict is None


class TestOtherErrorsKeepTheBuiltinVerdict:
    def test_real_content_refusal_is_not_retryable(self, plugin):
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            error_code="content_filter",
            message="Your prompt was rejected by the content policy.",
        )
        assert verdict is None

    def test_non_400_status_is_ignored(self, plugin):
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=503,
            error_code="content_filter",
            message=GUARDRAIL_MESSAGE.lower(),
        )
        assert verdict is None

    def test_unknown_error_code_is_ignored(self, plugin):
        verdict = plugin.classify_experientiallabs_api_error(
            None,
            status_code=400,
            error_code="invalid_request_error",
            message=GUARDRAIL_MESSAGE.lower(),
        )
        assert verdict is None

    def test_missing_message_is_ignored(self, plugin):
        verdict = plugin.classify_experientiallabs_api_error(
            None, status_code=400, error_code="content_filter"
        )
        assert verdict is None

    def test_malformed_body_never_raises(self, plugin):
        assert (
            plugin.classify_experientiallabs_api_error(
                None, status_code=400, body={"error": "not-a-dict"}
            )
            is None
        )
        assert (
            plugin.classify_experientiallabs_api_error(
                None, status_code=400, body=object()
            )
            is None
        )


class TestHermesWiring:
    def test_profile_exposes_the_hook(self, plugin):
        # Provider discovery layers $HERMES_HOME copies over this repo, so the
        # resolved lookup can be an older installed copy — assert on the profile
        # this module constructed.
        hook = getattr(plugin.experientiallabs, "classify_api_error", None)
        assert callable(hook)
        verdict = hook(
            None,
            status_code=400,
            error_code="content_filter",
            message=GUARDRAIL_MESSAGE.lower(),
        )
        assert verdict == {"reason": "content_policy_blocked", "retryable": True}

    def _classify(self, plugin, monkeypatch, code, message):
        import providers

        # Point the classifier at this module's profile so the test exercises
        # the repo code even when an installed copy shadows the lookup.
        monkeypatch.setattr(
            providers,
            "get_provider_profile",
            lambda name: plugin.experientiallabs if name == "experientiallabs" else None,
        )
        from agent.error_classifier import classify_api_error

        return classify_api_error(
            FakeBadRequest(_body(code, message)),
            provider="experientiallabs",
            model="glm-5.3-uncensored",
        )

    def test_hermes_classifier_honours_the_profile_hook(self, plugin, monkeypatch):
        classified = self._classify(
            plugin, monkeypatch, "unsupported_capability", INSPECTION_MESSAGE
        )

        assert classified.reason.value == "content_policy_blocked"
        assert classified.retryable is True

    def test_hermes_classifier_keeps_real_blocks_non_retryable(self, plugin, monkeypatch):
        refusal = "Your prompt was rejected by the content policy."
        # The plugin declines this message, so the non-retryable verdict below
        # is Hermes' built-in classification, not the hook's.
        assert (
            plugin.experientiallabs.classify_api_error(
                None,
                status_code=400,
                error_code="content_filter",
                message=refusal.lower(),
            )
            is None
        )

        classified = self._classify(plugin, monkeypatch, "content_filter", refusal)

        assert classified.retryable is False
