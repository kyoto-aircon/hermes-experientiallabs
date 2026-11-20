"""Experiential Labs provider profile for Hermes Agent.

Registers experientiallabs.ai as a first-class provider with feature parity
against Hermes' built-in Nous Portal provider wherever the gateway and
``ProviderProfile`` contract support it:

- OpenAI Chat Completions at ``https://api.experientiallabs.ai/v1`` and live
  catalog at ``GET /v1/models`` (filtering non-chat/non-tool rows, merging
  dynamic ``:free`` promo variants from ``free_models.py``, and seeding the
  in-memory + disk metadata cache in ``catalog.py`` at zero extra network cost).
- Reasoning control (``build_api_kwargs_extras``, ``supported_reasoning_efforts``,
  ``setup_reasoning_efforts``, ``reasoning_payload``): catalog-backed effort
  clamping and mandatory-reasoning disable omission (omit > HTTP 400).
- Conversation affinity & telemetry headers (``X-Explabs-Session-Id`` and
  ``X-Explabs-Tags``), plus per-request zero-data-retention enforcement
  (``provider: {"zdr": True}``) when ``data_use`` requires it.
- Live context bounds (``get_model_context_length`` + ``live_context_length``),
  per-model ``max_tokens_param``, and dynamic ``resolve_aux_model``.
- Cost attribution (``get_usage_cost``): ``:free`` allowance crediting ($0),
  BYOK pass-through reporting, inline ``usage.cost`` actual USD, and cached
  ``/v1/models`` nano-USD rate-card fallback (including cache reads/writes,
  1h cache writes, long-context tiers, and ``:fast`` 2x pricing).
- Account usage (``/usage``) via the Cost API in ``account_usage.py`` (wallet
  balance, key-cap reset cadences, depleted-credit warnings, spend rollups).
- Comprehensive gateway error classification (``classify_api_error``) covering
  transient guardrail 400s, retired ``:free`` routes, entitlements, region
  blocks, ZDR/policy refusals, billing/quota limits, and mandatory reasoning.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile

# Hermes imports the plugin as a package with submodule search locations, so the
# relative import is the normal path. Some tooling (pytest collecting this repo)
# imports this file without package context; fall back to a flat import then.
if __package__:
    from .account_usage import fetch_experientiallabs_account_usage
    from .catalog import (
        CatalogCapabilitiesView,
        _FREE_ALLOWANCE_NOTE,
        entry_supported_reasoning_efforts,
        estimate_cost_from_usage,
        fetch_model_context_live,
        fetch_v1_catalog,
        get_cached_entries,
        get_model_entry,
        invalidate_catalog_cache,
        resolve_reasoning_effort,
    )
    from .free_models import (
        _FREE_PROBE_TIMEOUT_S,
        fetch_free_suffix_slugs,
        merge_free_slugs,
    )
    from .sampling import resolve_temperature_override
else:  # pragma: no cover - only when imported without package context
    from account_usage import fetch_experientiallabs_account_usage
    from catalog import (
        CatalogCapabilitiesView,
        _FREE_ALLOWANCE_NOTE,
        entry_supported_reasoning_efforts,
        estimate_cost_from_usage,
        fetch_model_context_live,
        fetch_v1_catalog,
        get_cached_entries,
        get_model_entry,
        invalidate_catalog_cache,
        resolve_reasoning_effort,
    )
    from free_models import (
        _FREE_PROBE_TIMEOUT_S,
        fetch_free_suffix_slugs,
        merge_free_slugs,
    )
    from sampling import resolve_temperature_override

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
# Live fallback ladder consulted only when the catalog cache is warm and
# ``_AUX_MODEL`` is absent from it (prevents silent 404 rot on retired slugs).
_AUX_TEXT_CANDIDATES = (
    _AUX_MODEL,
    "deepseek-v4-flash",
    "gpt-6-luna",
    "qwen3.8-27b",
    "gemma-4-26b-a4b-it",
)
_AUX_VISION_CANDIDATES = (
    _AUX_MODEL,
    "gpt-6-luna",
    "claude-haiku-4.5",
    "gemini-4-neon",
)

_EXPLABS_HOST = "api.experientiallabs.ai"

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

_ENTITLEMENT_403_CODES = frozenset(
    {
        "model_not_granted",
        "model_blocked_for_org",
        "model_location_not_supported",
        "uncensored_access_suspended",
    }
)
_BILLING_402_CODES = frozenset(
    {
        "card_required",
        "email_unverified",
        "free_tier_requires_payment",
        "org_under_review",
        "promotion_ended",
    }
)
_BILLING_400_CODES = frozenset(
    {
        "capability_requires_purchase",
        "capability_requires_payment",
        "provider_account_quota",
    }
)
_MODEL_DEAD_400_CODES = frozenset(
    {
        "model_route_unavailable",
        "model_has_no_fast_path",
    }
)
_RATE_LIMIT_429_CODES = frozenset(
    {
        "rate_limit_exceeded",
        "admission_capacity_exceeded",
        "gateway_overloaded",
    }
)
_SERVER_502_CODES = frozenset({"all_routes_failed", "backend_unavailable"})
_OVERLOADED_503_CODES = frozenset(
    {"unavailable_route", "gateway_unavailable", "gateway_draining"}
)
_REASONING_DISABLE_MARKERS = ("none", "mandatory", "cannot", "disable")


def _is_explabs_url(base_url: str | None) -> bool:
    """True when *base_url* targets Experiential Labs (or is empty/default)."""
    raw = str(base_url or "").strip()
    if not raw:
        return True
    try:
        host = (urlparse(raw).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return host == _EXPLABS_HOST or host.endswith(f".{_EXPLABS_HOST}")


def _sticky_session_id(session_id: str | None) -> str | None:
    """Resolve the conversation-affinity scope, mirroring ``NousProfile``."""
    scope: str | None = None
    try:
        from agent.portal_tags import get_affinity_scope, get_conversation_context

        scope = get_affinity_scope() or get_conversation_context() or session_id
    except Exception:  # noqa: BLE001
        scope = session_id

    if not scope:
        return None

    try:
        from agent.transports.codex import _cache_scope_from_session_id

        hashed = _cache_scope_from_session_id(scope)
        if hashed:
            return hashed
    except Exception:  # noqa: BLE001
        pass

    cleaned = str(scope).strip()
    return cleaned or None


def _build_explabs_tags(session_id: str | None) -> dict[str, str]:
    """Safe, non-PII ``X-Explabs-Tags`` dictionary (<=16 entries, <=8 KiB).

    Extracts safe ``key=value`` metadata from Hermes' ``nous_portal_tags`` when
    available (skipping ``conversation`` because it travels in
    ``X-Explabs-Session-Id``).
    """
    tags: dict[str, str] = {"product": "hermes-agent"}
    try:
        from agent.portal_tags import nous_portal_tags

        raw_tags = nous_portal_tags(session_id=session_id)
        if isinstance(raw_tags, (list, tuple)):
            for item in raw_tags:
                if not isinstance(item, str) or "=" not in item:
                    continue
                k, v = item.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if not k or not v or k == "conversation":
                    continue
                if len(tags) < 16:
                    tags[k] = v[:128]
    except Exception:  # noqa: BLE001
        pass
    return tags


def classify_experientiallabs_api_error(
    error: BaseException | None,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str | None = None,
    body: object = None,
    model: str | None = None,
) -> dict[str, object] | None:
    """Classify Experiential Labs gateway errors before Hermes' built-in rules.

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

        inner = body.get("error") if isinstance(body, dict) else None
        code = (error_code or "").strip().lower()
        if not code and isinstance(inner, dict):
            code = str(inner.get("code") or "").strip().lower()

        err_type = ""
        if isinstance(inner, dict):
            err_type = str(inner.get("type") or "").strip().lower()
        elif isinstance(body, dict):
            err_type = str(body.get("type") or "").strip().lower()

        param = ""
        if isinstance(inner, dict):
            param = str(inner.get("param") or "").strip().lower()

        text = message or ""
        if not text and isinstance(inner, dict):
            text = str(inner.get("message") or "")
        if not text and isinstance(body, dict):
            text = str(body.get("message") or "")
        text_lower = text.lower()

        if status_code == 400:
            if code in _GUARDRAIL_ERROR_CODES and any(
                marker in text_lower for marker in _GUARDRAIL_MESSAGES
            ):
                return {"reason": "content_policy_blocked", "retryable": True}
            if code == "refusal":
                return {"reason": "content_policy_blocked", "retryable": False}
            if code in ("invalid_parameter", "unsupported_parameter"):
                # A rejection naming temperature is a sampling constraint, not a
                # reasoning control: decline so the built-in format_error verdict
                # stands (core's aux ladder strips temperature and remembers the
                # route; the main loop keeps its fallback). See plan-w1.md.
                if "temperature" in param or "temperature" in text_lower:
                    return None
                if ("reasoning" in param or "reasoning" in text_lower) and any(
                    m in text_lower for m in _REASONING_DISABLE_MARKERS
                ):
                    return {"reason": "reasoning_mandatory", "retryable": True}
            if code in _MODEL_DEAD_400_CODES:
                return {
                    "reason": "model_not_found",
                    "retryable": False,
                    "should_fallback": True,
                }
            if code in _BILLING_400_CODES:
                return {
                    "reason": "billing",
                    "retryable": False,
                    "should_fallback": True,
                }
            if code == "provider_credential_rejected":
                return {
                    "reason": "auth",
                    "retryable": False,
                    "should_fallback": True,
                }
            return None

        if status_code == 402 and code in _BILLING_402_CODES:
            return {
                "reason": "billing",
                "retryable": False,
                "should_fallback": True,
            }

        if status_code == 403:
            if code == "zdr_required":
                return {
                    "reason": "provider_policy_blocked",
                    "retryable": False,
                    "should_fallback": True,
                }
            if code == "needs_subscription":
                return {"reason": "provider_policy_blocked", "retryable": False}
            if code in _ENTITLEMENT_403_CODES:
                return {
                    "reason": "model_entitlement",
                    "retryable": False,
                    "should_fallback": True,
                }
            return None

        if status_code == 451 and code == "model_unavailable_in_region":
            return {
                "reason": "model_entitlement",
                "retryable": False,
                "should_fallback": True,
            }

        if status_code == 413 and code == "request_too_large":
            return {
                "reason": "payload_too_large",
                "retryable": False,
                "should_compress": True,
            }

        if status_code in (409, 500) and code == "idempotency_replay_unavailable":
            # Gateway docs: never auto-retry with a new Idempotency-Key (risk of double billing).
            return {"reason": "format_error", "retryable": False}

        if status_code == 429:
            if code == "insufficient_quota":
                return {
                    "reason": "billing",
                    "retryable": False,
                    "should_rotate_credential": True,
                    "should_fallback": True,
                }
            if code == "promo_credits_now":
                return {
                    "reason": "billing",
                    "retryable": False,
                    "should_fallback": True,
                }
            if code == "free_limit_reached":
                if err_type == "insufficient_quota":
                    return {
                        "reason": "billing",
                        "retryable": False,
                        "should_fallback": True,
                    }
                return {
                    "reason": "rate_limit",
                    "retryable": True,
                    "should_fallback": True,
                }
            if code in _RATE_LIMIT_429_CODES:
                return {"reason": "rate_limit", "retryable": True}
            if code == "unavailable_route":
                return {"reason": "overloaded", "retryable": True}
            return None

        if status_code == 502 and code in _SERVER_502_CODES:
            return {"reason": "server_error", "retryable": True}

        if status_code == 503:
            if code == "model_unavailable":
                return {
                    "reason": "model_not_found",
                    "retryable": False,
                    "should_fallback": True,
                }
            if code in _OVERLOADED_503_CODES:
                return {"reason": "overloaded", "retryable": True}
            if code == "deadline_exceeded":
                return {"reason": "timeout", "retryable": True}
            return None

        return None
    except Exception:  # pragma: no cover - a hook must never break classification
        return None


def _setup_reasoning_efforts(model: str) -> tuple[str, ...] | None:
    """Per-model effort choices for ``hermes model`` setup picker."""
    return entry_supported_reasoning_efforts(get_model_entry(model, warm=True))


class ExperientialLabsProfile(ProviderProfile):
    """Experiential Labs gateway — Nous-parity catalog, reasoning, cost, and /usage hooks."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Live ``/v1/models`` (seeded + tool-filtered) plus current ``:free`` variants.

        Fetches ``GET /v1/models`` once via ``fetch_v1_catalog`` (filtering out
        embeddings/TTS/non-tool entries and seeding the in-memory + disk
        metadata cache at zero extra network cost), then merges dynamic
        ``<slug>:free`` promotional variants from the keyless public catalog
        probe. Both probes fail open independently; with neither, Hermes falls
        back to ``fallback_models``.
        """
        live = fetch_v1_catalog(
            api_key=api_key, base_url=base_url, timeout=timeout
        )
        free = fetch_free_suffix_slugs(timeout=min(_FREE_PROBE_TIMEOUT_S, timeout))
        return merge_free_slugs(live, free)

    def invalidate_catalog_cache(self) -> None:
        """Drop in-process catalog caches on ``/model --refresh``."""
        invalidate_catalog_cache()

    def build_extra_body(
        self,
        *,
        session_id: str | None = None,
        data_use: frozenset[str] = frozenset(),
        **context: Any,
    ) -> dict[str, Any]:
        """Enforce per-request ZDR when an outgoing connector result requires it.

        Mirrors ``NousProfile.build_extra_body``: ``provider.zdr`` is sent ONLY
        when ``data_use`` is non-empty (e.g. ``zero_retention`` / ``no_training``),
        never unconditionally from config.
        """
        if data_use:
            return {"provider": {"zdr": True}}
        return {}

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Attach catalog-clamped ``reasoning_effort``, the sampling clamp, and
        session/tag headers.

        - Reasoning: uses the cached ``/v1/models`` capability map (never blocks
          on HTTP). Omits disable (``none`` / ``enabled: False``) on
          mandatory-reasoning routes and non-reasoning models; clamps positive
          efforts monotonically to the model's supported vocabulary.
        - Sampling: pinned routes (``minimum_temperature == maximum_temperature``)
          force the pinned top-level ``temperature`` (overrides the caller's via
          the transport's post-``_base_kwargs`` merge); published ranges clamp the
          caller's value when it is known. Cold cache, ``supports_temperature:
          false``, and reasoning-gated sampling leave the request untouched
          (``sampling.py`` / ``plan-w1.md``).
        - Attribution & conversation linking: attaches ``X-Explabs-Session-Id``
          (hashed by the gateway) and safe non-PII ``X-Explabs-Tags`` headers
          (stripped before forwarding upstream).
        """
        top_level: dict[str, Any] = {}

        entry = get_model_entry(model, warm=True)
        if isinstance(reasoning_config, dict) and reasoning_config:
            effort = resolve_reasoning_effort(model, reasoning_config, entry=entry)
            if effort is not None:
                top_level["reasoning_effort"] = effort

        # Sampling clamp (W1): a top-level temperature override wins over the
        # caller's value; None means "leave the request untouched" (cold cache,
        # supports_temperature=false, or sampling gated on reasoning being off).
        reasoning_disabled = isinstance(reasoning_config, dict) and (
            reasoning_config.get("enabled") is False
            or str(reasoning_config.get("effort") or "").strip().lower() == "none"
        )
        temperature = resolve_temperature_override(
            entry,
            context.get("temperature"),
            reasoning_disabled=reasoning_disabled,
        )
        if temperature is not None:
            top_level["temperature"] = temperature

        headers: dict[str, str] = {}
        sticky = _sticky_session_id(session_id)
        if sticky:
            headers["X-Explabs-Session-Id"] = sticky
        tags = _build_explabs_tags(session_id)
        if tags:
            headers["X-Explabs-Tags"] = json.dumps(tags, separators=(",", ":"))
        if headers:
            top_level["extra_headers"] = headers

        return {}, top_level

    def supported_reasoning_efforts(
        self, model: str | None
    ) -> tuple[str, ...] | None:
        """Tri-state reasoning effort vocabulary from cached ``/v1/models``."""
        return entry_supported_reasoning_efforts(get_model_entry(model, warm=True))

    def reasoning_payload(
        self,
        model: str | None,
        reasoning_config: dict | None,
        *,
        base_url: str,
    ) -> dict | None:
        """Endpoint-gated reasoning probe & payload (mirrors core's host gate)."""
        if not _is_explabs_url(base_url):
            return None
        entry = get_model_entry(model, warm=True)
        if reasoning_config is None:
            if isinstance(entry, dict) and entry.get("supports_reasoning") is True:
                default_eff = entry.get("default_reasoning_effort") or "medium"
                return {"effort": default_eff}
            return None
        effort = resolve_reasoning_effort(model, reasoning_config, entry=entry)
        return {"effort": effort} if effort is not None else None

    def get_model_context_length(self, model: str) -> int | None:
        """Cache-only context window from ``/v1/models`` (never blocks on HTTP)."""
        entry = get_model_entry(model, warm=True)
        if isinstance(entry, dict):
            ctx = entry.get("context_window")
            if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
                return ctx
        return None

    def live_context_length(
        self, model: str, *, api_key: str = "", base_url: str = ""
    ) -> int | None:
        """Live context bound from ``GET /api/models/<slug>`` (prefers ``max_input_tokens``)."""
        return fetch_model_context_live(model, api_key=api_key, base_url=base_url)

    def max_tokens_param(
        self, *, base_url: str, model: str = ""
    ) -> str | None:
        """Per-model max-tokens field from cached ``/v1/models`` (``chat_max_tokens_field``)."""
        if not _is_explabs_url(base_url):
            return None
        entry = get_model_entry(model, warm=False)
        if isinstance(entry, dict):
            field_name = entry.get("chat_max_tokens_field")
            if field_name in ("max_tokens", "max_completion_tokens"):
                return field_name
        return None

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """Cost attribution: ``:free`` allowance, BYOK, inline ``usage.cost``, or rate card."""
        entry = get_model_entry(model, warm=False)
        return estimate_cost_from_usage(model, usage, entry=entry)

    def fetch_account_usage(
        self, *, base_url: str | None = None, api_key: str | None = None
    ):
        return fetch_experientiallabs_account_usage(
            self.name, base_url=base_url, api_key=api_key
        )

    def get_max_tokens(self, model: str | None) -> int | None:
        """Per-model completion token ceiling from cached catalog (never blocks on HTTP)."""
        entry = get_model_entry(model, warm=False)
        if isinstance(entry, dict):
            max_out = entry.get("max_output_tokens")
            if isinstance(max_out, int) and not isinstance(max_out, bool) and max_out > 0:
                return max_out
        return self.default_max_tokens

    def resolve_aux_model(self, *, vision: bool = False) -> str:
        """Live cheap-tier aux pick from cached catalog, falling back to ``_AUX_MODEL``."""
        entries = get_cached_entries(warm=False)
        if entries:
            candidates = _AUX_VISION_CANDIDATES if vision else _AUX_TEXT_CANDIDATES
            for slug in candidates:
                entry = entries.get(slug)
                if not isinstance(entry, dict):
                    continue
                if vision and entry.get("supports_vision") is False:
                    continue
                return slug
        return _AUX_MODEL

    def default_vision_model(self) -> str | None:
        return self.resolve_aux_model(vision=True)


# Newer ProviderProfile fields added after the declared requires_hermes floor
# (v0.21.6): pass only those accepted by the installed Hermes dataclass so CI
# against v0.21.6 and installs on newer Hermes both work cleanly.
_OPTIONAL_PROFILE_FIELDS: dict[str, Any] = {
    "hostnames": (_EXPLABS_HOST,),
    "aggregator": True,
    "catalog_aliases": True,
    "setup_reasoning_efforts": _setup_reasoning_efforts,
    "replays_tool_call_signatures": True,
}
_SUPPORTED_PROFILE_FIELDS = getattr(ProviderProfile, "__dataclass_fields__", {})
_EXTRA_PROFILE_KWARGS = {
    key: value
    for key, value in _OPTIONAL_PROFILE_FIELDS.items()
    if key in _SUPPORTED_PROFILE_FIELDS
}

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
    model_capabilities=CatalogCapabilitiesView(),
    classify_api_error=classify_experientiallabs_api_error,
    **_EXTRA_PROFILE_KWARGS,
)

register_provider(experientiallabs)
