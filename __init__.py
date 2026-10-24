"""Experiential Labs provider profile for Hermes Agent.

Registers experientiallabs.ai as a first-class provider. The gateway speaks the
OpenAI Chat Completions API at ``https://api.experientiallabs.ai/v1`` and
lists its ~900 models at ``GET /v1/models``, so the profile's defaults cover
inference and the ``/model`` picker. Account usage (``/usage``) is served by
the Cost API in ``account_usage.py``.

Registering the profile is enough to wire it into Hermes' auth registry, setup
wizard, Desktop Settings → Providers ("keys" tab), ``/model`` picker, doctor
health check, and usage view — the standard provider-plugin path. Credentials
are declared through ``env_vars`` and edited in that native keys tab; the
manifest's ``config_schema`` only carries the ``/usage`` knobs.
"""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile

# Hermes imports the plugin as a package with submodule search locations, so the
# relative import is the normal path. Some tooling (pytest collecting this repo)
# imports this file without package context; fall back to a flat import then.
if __package__:
    from .account_usage import fetch_experientiallabs_account_usage
else:  # pragma: no cover - only when imported without package context
    from account_usage import fetch_experientiallabs_account_usage

# Curated /model picker fallback: cheap and free tiers that are all
# tool-calling capable. Live GET /v1/models supersedes this when reachable.
_FALLBACK_MODELS = (
    "qwen3.8-27b",
    "deepseek-v4-flash",
    "gpt-6-luna",
    "claude-haiku-4.5",
    "gpt-6-astra",
)

# Cheap, vision-capable tier for auxiliary work (titles, compression, vision).
_AUX_MODEL = "glm-5.3-flash"

# The hosted gateway runs a content-inspection/guardrail service in front of the
# models. That service intermittently fails closed, answering HTTP 400 before
# the request reaches a model: identical requests pass in one window and fail in
# the next (see AGENTS.md, "Known gateway quirks"). Hermes' built-in classifier
# reads a ``content_filter`` 400 as a deterministic content-policy block and
# ends the turn, so the profile marks exactly these two gateway-service
# signatures retryable and Hermes backs off and retries. Real content refusals
# carry different messages and keep the built-in verdict.
#
# Lowercase on purpose: Hermes passes this hook a lowercased message
# (``_Ctx.msg`` in agent/error_classifier.py).
_GUARDRAIL_ERROR_CODES = frozenset({"unsupported_capability", "content_filter"})
_GUARDRAIL_MESSAGES = (
    "content inspection does not support this complete request",
    "a gateway guardrail could not complete this request",
)


def classify_experientiallabs_api_error(
    error: BaseException,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str | None = None,
    body: object = None,
    model: str | None = None,
) -> dict[str, object] | None:
    """Mark the gateway's transient content-inspection 400s as retryable.

    Hermes calls this through ``ProviderProfile.classify_api_error`` before its
    built-in status handling. Returning ``None`` leaves the built-in verdict in
    place; a hook must never raise into the classifier.
    """
    try:
        if status_code != 400:
            return None
        code = (error_code or "").strip().lower()
        text = message or ""
        inner = body.get("error") if isinstance(body, dict) else None
        if not code and isinstance(inner, dict):
            code = str(inner.get("code") or "").strip().lower()
        if not text and isinstance(inner, dict):
            text = str(inner.get("message") or "")
        if not text and isinstance(body, dict):
            text = str(body.get("message") or "")
        if code not in _GUARDRAIL_ERROR_CODES:
            return None
        if not any(marker in text.lower() for marker in _GUARDRAIL_MESSAGES):
            return None
        return {"reason": "content_policy_blocked", "retryable": True}
    except Exception:  # pragma: no cover - a hook must never break classification
        return None


class ExperientialLabsProfile(ProviderProfile):
    """Experiential Labs gateway — Cost API backs the /usage snapshot."""

    def fetch_account_usage(self, *, base_url: str | None = None,
                            api_key: str | None = None):
        return fetch_experientiallabs_account_usage(
            self.name, base_url=base_url, api_key=api_key
        )

    def default_vision_model(self) -> str | None:
        return _AUX_MODEL


experientiallabs = ExperientialLabsProfile(
    name="experientiallabs",
    aliases=("explabs", "experiential-labs"),
    display_name="Experiential Labs",
    description="Experiential Labs — OpenAI-compatible multi-model gateway",
    signup_url="https://platform.experientiallabs.ai/settings/api-keys",
    # Order matters: the first var is the primary key, and the trailing
    # *_BASE_URL entry is picked up by Hermes as the base-URL override.
    env_vars=("EXPLABS_API_KEY", "EXPERIENTIALLABS_API_KEY", "EXPLABS_BASE_URL"),
    base_url="https://api.experientiallabs.ai/v1",
    auth_type="api_key",
    supports_health_check=True,
    supports_vision=True,
    default_aux_model=_AUX_MODEL,
    fallback_models=_FALLBACK_MODELS,
    classify_api_error=classify_experientiallabs_api_error,
)

register_provider(experientiallabs)
