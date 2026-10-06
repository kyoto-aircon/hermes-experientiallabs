"""Contract tests for the Experiential Labs provider profile.

Pins identity, credential wiring, and the curated model defaults without any
network access.
"""

from __future__ import annotations

import pytest

KNOWN_FALLBACKS = {
    "qwen3.8-27b",
    "deepseek-v4-flash",
    "gpt-6-luna",
    "claude-haiku-4.5",
    "gpt-6-astra",
}


class TestRegistration:
    def test_profile_registers(self, profile):
        from providers.base import ProviderProfile

        assert isinstance(profile, ProviderProfile)

    def test_canonical_name_resolves(self, profile):
        import providers

        assert providers.get_provider_profile("experientiallabs") is profile

    @pytest.mark.parametrize("alias", ["explabs", "experiential-labs"])
    def test_aliases_resolve_to_same_profile(self, profile, alias):
        import providers

        assert providers.get_provider_profile(alias) is profile

    def test_register_provider_is_idempotent(self, profile):
        import providers

        providers.register_provider(profile)
        assert providers.get_provider_profile("experientiallabs") is profile


class TestIdentity:
    def test_name_and_display(self, profile):
        assert profile.name == "experientiallabs"
        assert profile.display_name == "Experiential Labs"
        assert profile.description
        assert profile.signup_url.startswith("https://")

    def test_base_url_and_auth(self, profile):
        assert profile.base_url == "https://api.experientiallabs.ai/v1"
        assert profile.auth_type == "api_key"
        assert profile.api_mode == "chat_completions"

    def test_hostname_derived_from_base_url(self, profile):
        assert profile.get_hostname() == "api.experientiallabs.ai"

    def test_vision_enabled(self, profile):
        assert profile.supports_vision is True


class TestCredentials:
    def test_env_vars_primary_first(self, profile):
        assert profile.env_vars[0] == "EXPLABS_API_KEY"
        assert "EXPERIENTIALLABS_API_KEY" in profile.env_vars

    def test_env_vars_exact_order_and_contents(self, profile):
        assert profile.env_vars == (
            "EXPLABS_API_KEY",
            "EXPERIENTIALLABS_API_KEY",
            "EXPLABS_BASE_URL",
        )

    def test_health_check_enabled(self, profile):
        assert profile.supports_health_check is True


class TestModelDefaults:
    def test_aux_model_is_cheap_flash_tier(self, profile):
        assert profile.default_aux_model == "glm-5.3-flash"

    def test_vision_model_uses_aux(self, profile):
        assert profile.default_vision_model() == "glm-5.3-flash"

    def test_fallback_models_are_the_curated_cheap_set(self, profile):
        assert set(profile.fallback_models) == KNOWN_FALLBACKS
        assert len(profile.fallback_models) == len(set(profile.fallback_models))

    def test_fallback_models_do_not_include_aux(self, profile):
        assert profile.default_aux_model not in profile.fallback_models


class TestHooks:
    def test_fetch_models_uses_default_openai_listing(self, profile):
        from providers.base import ProviderProfile

        # The gateway serves GET /v1/models in standard OpenAI shape, so the
        # base implementation must not be shadowed.
        assert type(profile).fetch_models is ProviderProfile.fetch_models

    def test_account_usage_hook_is_overridden(self, profile):
        from providers.base import ProviderProfile

        assert type(profile).fetch_account_usage is not ProviderProfile.fetch_account_usage

    def test_fetch_account_usage_delegates_with_name_and_credentials(
        self, plugin, profile, monkeypatch
    ):
        captured = {}
        sentinel = object()

        def fake_fetch(name, *, base_url=None, api_key=None):
            captured.update(name=name, base_url=base_url, api_key=api_key)
            return sentinel

        monkeypatch.setattr(plugin, "fetch_experientiallabs_account_usage", fake_fetch)
        result = profile.fetch_account_usage(base_url="https://example.test/v1", api_key="tok")

        assert result is sentinel
        assert captured == {
            "name": "experientiallabs",
            "base_url": "https://example.test/v1",
            "api_key": "tok",
        }
