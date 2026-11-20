# W3 — retry safety: findings & shipped scope

Branch `feat/retry-safety`. Part A (429 reset hints) shipped; Part B
(Idempotency-Key) is **findings + default-off design only** — no provably safe
injection point exists in Hermes today, so no key is ever sent.

## Part A — shipped: 429 reset hints on classify verdicts

### What the core actually consumes (verified against the checkout)

- `agent/error_classifier.py::_profile_verdict` (~line 850) runs the profile's
  `classify_api_error` **before** the built-in pipeline and copies a dict
  `error_context` from the verdict onto the `ClassifiedError` (lines 886–887).
  The built-in `_status_429` (~line 1262) attaches
  `error_context={"reset_at": time.time() + reset}` — **absolute epoch
  seconds** — but it never runs when the plugin verdict wins. That was the
  gap: plugin-classified gateway 429s reached the retry-exhausted terminal
  copy with an empty context.
- `agent/turn_recovery.py::max_retries_exhausted_result` (~line 1301) reads
  `classified.error_context.get("reset_at")` and computes
  `reset_seconds = reset_at - time.time()` for `exhausted_copy` — epoch
  seconds, not a relative delay.
- `agent/nous_rate_guard.py::record_nous_rate_limit` reads
  `error_context["reset_at"]` as `isinstance(ctx_reset, (int, float)) and
  ctx_reset > now` (epoch seconds). Its only caller
  (`turn_recovery.py::_is_genuine_nous_rate_limit`, ~line 1980) is gated on
  `agent.provider == "nous"`, so Experiential Labs verdicts never trip the
  Nous cross-session breaker — the shape is mirrored for compatibility, not
  because that path runs for this provider.
- `agent/turn_api_error.py` / `route_classified_error` /
  `settle_unrecovered_error` pass `reset_at` from the **turn's own**
  `extract_api_error_context(api_error)` (raw-error extraction in
  `agent/agent_runtime_helpers.py`, ~line 3622) into
  `_try_activate_fallback` → `agent/fallback_cooldown.py`, whose
  `_parse_absolute_timestamp` accepts epoch s / epoch ms / ISO-8601 and
  ignores expired/invalid values.
- Retry **backoff** needs no plugin help: `compute_error_backoff` →
  `agent/retry_utils.py::provider_retry_after_seconds` reads `Retry-After`
  (both casings) straight off `error.response.headers`, then a body
  `retry_after` field. `agent/rate_limit_tracker.py` only parses
  `x-ratelimit-*` headers off *successful* responses for `/usage` display.

### Implementation

- New pure module `retry_hints.py` (stdlib only at load, no I/O, never
  raises): `reset_epoch_from_error` / `rate_limit_error_context`. Sources in
  priority order, mirroring core's `extract_api_error_context`: body
  `resets_at`/`reset_at` (epoch s/ms, ISO-8601) → body `resets_in_seconds` →
  body `retry_after` → header `Retry-After` (seconds or HTTP-date) →
  OpenAI-style `x-ratelimit-reset-{requests,tokens}[-1h]` relative windows
  (hourly first, mirroring `nous_rate_guard._parse_reset_seconds`) →
  `x-ratelimit-reset` epoch → message grammars (`retry after N s`,
  `resets in Xhr Ymin`). Defensive header normalisation (dict, httpx-style
  case-insensitive mappings, `.get`-only objects, hostile objects). Expired,
  zero, non-finite and >366-day values are rejected as malformed.
- `__init__.py`: surgical 429-region edit only. Retryable verdicts
  (`rate_limit_exceeded`, `admission_capacity_exceeded`, `gateway_overloaded`,
  hourly/daily `free_limit_reached`, 429 `unavailable_route`) gain
  `error_context={"reset_at": <epoch s>}` when a signal exists; billing
  verdicts and every other status are byte-identical to before. No hint →
  verdict unchanged (fail open).
- Gateway contract grounding (llms.txt): the three 429 rate-limit codes carry
  `Retry-After` seconds; `free_limit_reached` messages also name the reset.
  `x-ratelimit-*` headers are not documented gateway behaviour — parsed
  defensively as a secondary source only.
- Tests: `tests/test_retry_safety.py` — fake SDK-shaped errors
  (`.status_code`/`.body`/`.response.headers`), header/body/message sources,
  precedence, malformed/hostile inputs, billing-verdict invariance, and
  end-to-end wiring through the real `classify_api_error` including a
  round-trip through `fallback_cooldown._provider_reset_delay` (the actual
  downstream parser). All offline.

## Part B — Idempotency-Key: NOT shipped (no provably safe seam)

Gateway contract (llms.txt, verified): honored on POST `/v1/chat/completions`
and `/v1/responses` only (NOT embeddings/messages). Same key + identical body
replays the held response; the hold is **serving-worker memory only** — not
guaranteed across restart, eviction or catalog revision; within the dedup
scope an un-replayable duplicate is *refused* (`idempotency_replay_unavailable`
409/500), not re-executed. Same key + different body → 409
`idempotency_conflict`. A new key starts a new operation and **can charge
again**. Keyed responses carry **no `usage.cost`** (original or replay).

### Seams examined

1. **`ProviderProfile.request_headers`** (`providers/base.py:480`) — pure and
   memoized per argument set by `agent/provider_route_hooks.py`
   (`_PURE_HOOKS`, memo key = registry generation + profile + frozen
   `(base_url, model, user_initiated)`). No turn/request identity, no content
   → cannot vary per operation. Additionally
   `agent/turn_api_request.py::_apply_user_turn_request_headers` fires it
   **only on the first user-initiated attempt of a turn** (flag consumed
   once); retries get nothing. Doubly disqualified.
2. **`build_api_kwargs_extras` / `build_extra_body`** — receive
   `session_id`/`cache_scope_id`, model, reasoning config, `data_use`; no
   message content, no turn/api-request identity. A session-scoped key would
   collide across every operation in the session → 409 on the second call.
   (Also owned by W1's temperature work — untouched here.)
3. **`llm_request` middleware** (`hermes_cli/middleware.py::apply_llm_request_middleware`,
   called per attempt from `agent/turn_api_request.py::build_request_payload`
   with `task_id/turn_id/api_request_id/api_call_count/session_id/model/
   provider/base_url/api_mode` and the full `api_kwargs`) — the only seam
   that sees both request identity and content, and it can rewrite
   `extra_headers`. Two independent blockers:
   - **Registration is unreachable from this plugin.**
     `hermes_cli/plugins_discovery.py:309` skips `kind: model-provider`
     ("handled by providers/ discovery"); `providers/__init__.py` imports the
     module solely for `register_provider`. The PluginManager lifecycle that
     exposes `PluginContext.register_middleware`
     (`hermes_cli/plugins_context.py:757`) never runs for this plugin — the
     same reason `_profile_verdict` exists for classify. Registering
     middleware would require a second plugin component of another kind
     (separate install/enable lifecycle) — out of scope for a model-provider.
   - **No key derivation is provably safe even with the seam.**
     `api_request_id = f"{turn_id}:api:{api_call_count}"`
     (`agent/conversation_loop.py:1771`) is set once before
     `_run_api_retry_loop` and is stable across retries of one logical call
     ✓ — but `apply_retry_restarts`
     (`agent/turn_iteration_prep.py:437+`) **refunds `api_call_count -= 1`**
     on redirect, compression and fallback-rebuild restarts, re-issuing the
     *same* id with a different body (and possibly a different
     provider/model) → same key + different body → 409. And the body is not
     byte-stable across retries even inside the retry loop: image shrink,
     reasoning-disable strip, output-cap/affordable-budget clamps, and
     per-attempt prompt-cache re-decoration
     (`build_request_payload` re-runs `_redecorate_prompt_cache_for_provider`
     every attempt) all mutate it → 409 with a stable key.
     Content-hash keys (stable iff body identical) fail the other way:
     identical bodies across *distinct* operations (refunded-count reissues,
     repeated aux/"continue" calls) replay a stale held response — including
     stale `tool_call` ids, corrupting the transcript; empty-content retries
     would replay the same empty response, defeating their purpose; and any
     body-mutating retry after a server-processed-but-lost response mints a
     new key → double-bill. Hermes offers no seam guaranteeing byte-identical
     retries, so neither key family is safe.
4. **`pre_api_request` hook** — observability payload
   (`_api_request_payload_for_hook`), also PluginManager-only; no mutation
   path and no registration for this kind.

### Residual gateway-side limits (even with a perfect seam)

- Worker-local hold: after eviction/restart/catalog revision, replay is not
  guaranteed; the dedup fence refuses rather than re-executes *within scope*,
  but cross-revision protection "requires the deployment's operation-identity
  fence" — the key alone is not a double-bill guarantee.
- Keyed responses lose `usage.cost` → `get_usage_cost` would fall back to
  rate-card estimates on every keyed call, degrading billing accuracy.
- `idempotency_replay_unavailable` must stay non-retryable (already classified
  as `format_error`, retryable=False) — a new key there can execute and
  charge again.

### Default-off design (future, requires core seams first)

Prerequisites — none exist today, all are Hermes-core changes:

1. A true **operation identity** per logical request that is never refunded
   or reused across restarts (today's `api_request_id` fails this) and is
   exposed to a per-attempt seam together with the final wire body.
2. A **byte-identical-retry guarantee** (or gateway-side body-hash dedup):
   recovery paths that mutate the body must mint a new operation identity,
   and transport-level retries of an uncertain outcome must not.
3. A registration path for model-provider plugins into the middleware
   lifecycle, or a new pure per-attempt profile hook
   (`operation_headers(operation_id, body_digest)`).

Then, and only then:

- `config_schema` knob `idempotency_keys: bool, default false` (non-secret).
- Key = `uuid5(fixed-namespace, f"{session_id}:{operation_id}").hex` —
  stable across retries of one operation, unique across operations; attached
  via `extra_headers` on chat/responses routes only, never embeddings, never
  aux/title calls (they run through a separate client path anyway).
- 409 `idempotency_conflict` → non-retryable verdict naming the original
  request (check usage/history); `idempotency_replay_unavailable` keeps the
  existing non-retryable verdict. No automatic new-key retry, ever.
- Accept and document the `usage.cost` loss while keyed (rate-card fallback).
- Live verification budget at that time: ≤6 tiny `gpt-6-luna` completions,
  `live`-marked only (replay, conflict, eviction behaviour).

Shipping even default-off code now would be dead code (no registration seam)
plus a misleading settings knob; AGENTS.md forbids speculative settings.
Findings only, per the plan's "otherwise deliver findings and a default-off
design, not a double-bill risk".

## Residual risks (shipped scope)

- Hint honesty depends on the gateway's `Retry-After`/reset fields; a silent
  gateway (no signal) yields the unchanged verdict — same behaviour as before.
- The hint improves the terminal copy and any future `reset_at` consumer of
  the classified context; backoff and fallback cooldowns already read the raw
  error independently, so no timing behaviour changes.
- 366-day sanity bound rejects absurd resets; a legitimate >1-year window
  (none documented for rate limits) would lose its hint, not its verdict.
