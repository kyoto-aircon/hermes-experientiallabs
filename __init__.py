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

Free models come in two kinds, credited differently (see ``free_models.py``):
stable $0 catalog routes, and dynamic ``<slug>:free`` promotional variants
that draw on a separate free allowance instead of wallet credits. The picker
merges the current ``:free`` variants from the keyless public catalog's
promotions on every live fetch, ``get_usage_cost`` reports ``:free`` turns as
$0 wallet spend, and a 404 on a ``:free`` route (an expired promotion) is
classified as a retired model so Hermes falls back instead of retrying.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Hermes imports the plugin as a package with submodule search locations, so the
# relative import is the normal path. Some tooling (pytest collecting this repo)
# imports this file without package context; fall back to a flat import then.
if __package__:
    from .account_usage import fetch_experientiallabs_account_usage
    from .free_models import (
        _FREE_PROBE_TIMEOUT_S,
        fetch_free_suffix_slugs,
        merge_free_slugs,
    )
else:  # pragma: no cover - only when imported without package context
    from account_usage import fetch_experientiallabs_account_usage
    from free_models import (
        _FREE_PROBE_TIMEOUT_S,
        fetch_free_suffix_slugs,
        merge_free_slugs,
    )

# Curated /model picker fallback: cheap and free tiers that are all
# tool-calling capable. Live GET /v1/models supersedes this when reachable.
# gemma-4-26b-a4b-it is the stable $0 tier (host-managed, zero-priced,
# tool-capable). Promotional ``:free`` variants are deliberately NOT pinned
# here: they expire with their promotions, so they enter the picker only
# through the live promotions probe (free_models.py), never a stale slug.
_FALLBACK_MODELS = (
    "qwen3.8-27b",
    "deepseek-v4-flash",
    "gpt-6-luna",
    "claude-haiku-4.5",
    "gpt-6-astra",
    "gemma-4-26b-a4b-it",
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

# ``:free`` turns draw on the gateway's promotional free allowance (hourly /
# daily caps), not wallet credits — the same split the built-in Nous provider
# makes for its free tier. Report them as $0 so local cost attribution never
# charges the wallet for an allowance-billed turn.
_FREE_ALLOWANCE_NOTE = (
    ":free routes draw on the gateway's promotional free allowance "
    "(hourly/daily caps), not wallet credits."
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
        # A 404 on a ``:free`` variant means the promotion retired the route.
        # Like Nous' retired free routes, the slug is dead for every credential,
        # so fall back instead of burning retries. Paid models keep the built-in
        # 404 handling (message-pattern based).
        if status_code == 404 and isinstance(model, str) and model.endswith(":free"):
            return {"reason": "model_not_found", "should_fallback": True}
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

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Live ``/v1/models`` plus the gateway's current ``:free`` variants.

        The suffix routes are dynamic (promotions come and go) and the
        key-scoped ``/v1/models`` list may omit them, so they are derived from
        the keyless public catalog's promotions array on every fetch and
        merged in, deduplicated. The probe fails open: without it the live
        list stands alone, and with neither, Hermes uses ``fallback_models``.
        """
        # Dict-unpacked kwargs: same call, keeps the credential passthrough
        # explicit for the base OpenAI-compatible listing.
        live = super().fetch_models(
            **{"api_key": api_key, "base_url": base_url, "timeout": timeout}
        )
        free = fetch_free_suffix_slugs(timeout=min(_FREE_PROBE_TIMEOUT_S, timeout))
        return merge_free_slugs(live, free)

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """``:free`` turns are allowance-billed, not wallet-billed — report $0.

        Mirrors the built-in Nous provider's free-tier crediting: the ``:free``
        suffix is the canonical free-SKU marker, so local cost attribution
        must not charge wallet credits for it. Paid routes return ``None`` and
        keep Hermes' built-in pricing path. Fails open if the cost layer is
        unavailable.
        """
        if not isinstance(model, str) or not model.endswith(":free"):
            return None
        try:
            from agent.usage_pricing import CostResult
        except Exception:  # noqa: BLE001 - cost layer must fail open
            return None
        return CostResult(
            amount_usd=Decimal("0"),
            status="included",
            source="none",
            label="free allowance",
            notes=(_FREE_ALLOWANCE_NOTE,),
        )

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
