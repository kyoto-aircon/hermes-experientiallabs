"""Contract tests for the plugin manifest and its ``config_schema``.

Pins the settings-page surface (Hermes #64165): keys, types and defaults —
both remaining fields are the ``/usage`` knobs, because provider credentials are
edited through Hermes' native Settings → Providers → API keys tab. Uses Hermes'
own manifest loader so the test fails when the manifest drifts from what Hermes
actually parses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def manifest():
    from hermes_cli.plugins_manifest import parse_manifest_file

    parsed = parse_manifest_file(
        REPO_ROOT / "plugin.yaml", REPO_ROOT, source="test", prefix=""
    )
    assert parsed is not None, "plugin.yaml must parse"
    return parsed


class TestManifestFields:
    def test_identity(self, manifest):
        assert manifest.name == "hermes-experientiallabs"
        assert manifest.kind == "model-provider"
        assert manifest.version == "0.4.0"
        assert manifest.description
        assert manifest.author == "Spheraz"

    def test_public_metadata(self, manifest):
        assert manifest.license == "MIT"
        assert manifest.homepage == "https://github.com/kyoto-aircon/hermes-experientiallabs"
        assert manifest.tags == ["experientiallabs", "model-provider", "openai-compatible"]

    def test_requires_hermes_floor_is_settings_ui_release(self, manifest):
        # The config_schema settings pages shipped in v0.21.6, and so did the
        # ProviderProfile.classify_api_error field this profile passes; the
        # floor must never be newer than the current Hermes release (catalog
        # rule 14).
        assert manifest.requires_hermes == ">=0.21.6"

    def test_no_requires_env_gate(self, manifest):
        # requires_env gates plugin loading; a provider must be discoverable
        # before any key exists (wizard flow sets it later).
        assert manifest.requires_env == []

    def test_pyproject_version_stays_in_sync(self, manifest):
        # AGENTS.md "Version sync": the pyproject [project] version only
        # exists because Hermes' plugin manager copies the file into a uv
        # workspace; it must never drift from the manifest.
        import tomllib

        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            project = tomllib.load(fh)["project"]
        assert project["version"] == manifest.version


class TestConfigSchema:
    def test_keys_and_types(self, manifest):
        # Credentials are not settings: EXPLABS_API_KEY and EXPLABS_BASE_URL are
        # declared via env_vars and edited in Hermes' native Providers → API keys
        # tab. Only the two /usage knobs belong to the plugin settings page.
        assert set(manifest.config_schema) == {
            "cost_api_base",
            "usage_timeout_s",
        }
        assert manifest.config_schema["cost_api_base"]["type"] == "str"
        assert manifest.config_schema["usage_timeout_s"]["type"] == "float"

    def test_no_secret_fields_duplicate_the_native_key_tab(self, manifest):
        # A secret field would be a second, unsynced editor of the same .env
        # line the Providers → API keys tab owns.
        assert not [
            key
            for key, spec in manifest.config_schema.items()
            if str(spec.get("type")).lower() == "secret"
        ]

    def test_defaults(self, manifest):
        assert (
            manifest.config_schema["cost_api_base"]["default"]
            == "https://api.experientiallabs.ai/api/v1"
        )
        assert manifest.config_schema["usage_timeout_s"]["default"] == 4.0

    def test_manifest_defaults_match_the_module_fallbacks(self, manifest, account_usage):
        # The settings pages render the manifest defaults while the usage
        # hook falls back to the module constants; drift would show users one
        # value while the fetch quietly used another.
        schema = manifest.config_schema
        assert schema["cost_api_base"]["default"] == account_usage._DEFAULT_COST_API_BASE
        assert account_usage._cost_urls(schema["cost_api_base"]["default"]) == (
            account_usage._CREDITS_URL,
            account_usage._KEY_URL,
        )
        assert schema["usage_timeout_s"]["default"] == account_usage._HTTP_TIMEOUT_S

    def test_every_field_documented_and_optional(self, manifest):
        # The settings page is the user-facing surface: every key needs a
        # label and description, and none may be required.
        for key, spec in manifest.config_schema.items():
            assert spec.get("label"), f"{key} must carry a label"
            assert spec.get("description"), f"{key} must carry a description"
            assert not spec.get("required"), f"{key} must be optional"

    def test_types_are_known_to_hermes(self, manifest):
        from hermes_cli.plugins_manifest import _CONFIG_SCHEMA_TYPES

        for key, spec in manifest.config_schema.items():
            assert str(spec.get("type")).lower() in _CONFIG_SCHEMA_TYPES, key

    def test_schema_with_empty_settings_yields_no_warnings(self, manifest):
        # A fresh install has no settings yet; that must be warning-free.
        from hermes_cli.plugins_manifest import validate_config_schema

        warnings = validate_config_schema(manifest.name, manifest.config_schema, {})
        assert warnings == []

    def test_settings_roundtrip_type_warnings(self, manifest):
        # Non-secret settings are type-checked by Hermes on save; feeding the
        # same check garbage values must produce the documented warnings.
        from hermes_cli.plugins_manifest import validate_config_schema

        warnings = validate_config_schema(
            manifest.name, manifest.config_schema, {"usage_timeout_s": "not-a-float"}
        )
        assert warnings == [
            "plugins.entries.hermes-experientiallabs.settings.usage_timeout_s "
            "should be float (got str)"
        ]

    def test_cost_api_base_type_warning(self, manifest):
        # The other non-secret field must produce the same warning shape.
        from hermes_cli.plugins_manifest import validate_config_schema

        warnings = validate_config_schema(
            manifest.name, manifest.config_schema, {"cost_api_base": {"not": "a str"}}
        )
        assert warnings == [
            "plugins.entries.hermes-experientiallabs.settings.cost_api_base "
            "should be str (got dict)"
        ]

    def test_valid_typed_settings_produce_no_warnings(self, manifest):
        # An int satisfies the float field (Hermes' float accepts int), and a
        # str satisfies cost_api_base: a clean settings block stays warning-free.
        from hermes_cli.plugins_manifest import validate_config_schema

        warnings = validate_config_schema(
            manifest.name,
            manifest.config_schema,
            {"cost_api_base": "https://mirror.test/api/v1", "usage_timeout_s": 4},
        )
        assert warnings == []
