# W1 — sampling clamp: findings & residual limitations

Branch `feat/sampling-clamp`, based on main `6c007c6`. Offline suite green
(322 passed, 1 live deselected; 49 new tests in `tests/test_sampling.py`).

## What shipped

- `catalog.py::parse_v1_model_item` now carries `supports_temperature`,
  `minimum_temperature`, `maximum_temperature`,
  `sampling_requires_reasoning_none` when present. Bounds parse as finite
  floats (bool/NaN/inf/garbage dropped, `0.0` kept); flags must be real bools.
  Absent stays absent, so cold/stale caches fail open.
- New pure module `sampling.py::resolve_temperature_override(entry,
  caller_temperature, *, reasoning_disabled)`:
  - pinned route (`min == max`) → the pinned value, whatever the caller asked;
  - published range → caller's value clamped, **only when known**;
  - `supports_temperature: false`, cold cache, contradictory bounds, unknown
    caller on a ranged route → `None` (request untouched).
- `ExperientialLabsProfile.build_api_kwargs_extras` emits the result as
  top-level `temperature`. Core applies top-level profile kwargs AFTER
  `_base_kwargs` (`agent/transports/chat_completions.py:671`,
  `agent/auxiliary_client.py:6565`), so the override beats the caller's value;
  proven offline by `TestTransportOverrideOrdering` driving the real transport.
- `sampling_requires_reasoning_none` gate: when true and reasoning is not known
  to be disabled (`enabled: False` or `effort: "none"` — same disable rule as
  `resolve_reasoning_effort`), no override is emitted. Adding a temperature
  while reasoning may be active would itself 400 on those lanes.
- Classify guard: a 400 `invalid_parameter`/`unsupported_parameter` naming
  `temperature` now declines (returns `None`) BEFORE the reasoning-wording
  branch, so wording like "temperature requires reasoning effort none" cannot
  misroute into the `reasoning_mandatory` recovery rung (wrong remedy: it drops
  reasoning, not temperature).

## Live catalog snapshot (keyed GET /v1/models, 2026-11-20)

314 rows, 309 carry `supports_temperature`:

- **7 pinned** (`min == max == 1.0`, all `sampling_requires_reasoning_none:
  false`): claude-fable-5.2, claude-haiku-5.5, claude-opus-5, claude-opus-5.5,
  claude-opus-latest, claude-sonnet-5.5, claude-sonnet-latest. (Plan said 9 —
  catalog drift; everything is derived from the cache, nothing hardcoded.)
- **227 ranged** (mostly `[0, 2]`, Claude 4.x `[0, 1]`).
- **73 `supports_temperature: false`** (embeddings/TTS/image routes plus
  claude-fable-latest, claude-haiku-latest, gpt-5.1-codex family, …).
- **11 `sampling_requires_reasoning_none: true`**: gpt-5.1-codex{,-max,-mini}
  (also `supports_temperature: false`) and gpt-5.4/5.5/5.6-{luna,sol,terra}/
  gpt-6-{luna,sol} ranged `[0, 2]` lanes — the reason for the gate above.

## Strip-and-retry research (task item 3)

**Verdict: no new retry verdict shipped — none is safe on the main path, and
the aux path already has one in core.**

### Main turn path — no safe mechanism

- `ProviderProfile.fixed_temperature` is a static per-profile field;
  `OMIT_TEMPERATURE` is consumed in `_base_kwargs` BEFORE the hook runs, so a
  per-model omission cannot be expressed from `build_api_kwargs_extras`.
  Top-level hook kwargs can override a field, never delete it.
- The recovery ladder (`agent/turn_recovery.py`, `agent/turn_api_error.py`) is
  keyed per `FailoverReason` with hardcoded remedies (image shrink, reasoning
  disable drop/floor, role merge, …). There is **no temperature rung and no
  generic "strip field X" hint** in the classify-verdict contract
  (`reason` + `retryable`/`should_compress`/`should_rotate_credential`/
  `should_fallback` + a few consumed `error_context` keys).
- Returning `retryable: True` for a deterministic temperature 400 would re-send
  the byte-identical body `max_retries` times, then fall back — burning paid
  retries for nothing. So temperature 400s keep the built-in `format_error`
  verdict (non-retryable → fallback chain), and the hook declines them.

### Aux path — core already strips and remembers (no plugin work needed)

`agent/auxiliary_client.py` has a parameter-repair ladder:

- `_parameter_rungs` temperature rung matches `_is_unsupported_parameter_error
  (exc, "temperature")` — the field name plus `UNSUPPORTED_PARAM_MARKERS`,
  which include the gateway's `unsupported_parameter` code and "not supported"
  wording (`invalid_parameter` bodies match via their message text).
- `_without_temperature` strips the key (returns `None` when it was not sent,
  so an unchanged request is never re-sent), retries once, and on success
  `remember_temperature_rejection` records `(route, bare model)`;
  `_fixed_temperature_for_model` then returns `OMIT_TEMPERATURE` for that
  route+model on every later aux call in the process.
- The ladder runs whenever the verdict is not a policy refusal
  (`auxiliary_error_policy.resend_cannot_help`); the built-in `format_error`
  verdict qualifies. This is why the plugin must keep DECLINING temperature
  400s — the new guard above protects exactly that.
- Interaction with the clamp: after a rejection memory strips temperature, a
  warm-cache pin re-adds `temperature=1.0` via `projection.top_level` on the
  next aux call — safe, since 1.0 is the only accepted value on pinned routes.

### Residual limitations (documented, not fixed)

1. **Main-path explicit temperature on `supports_temperature: false` routes**
   still 400s → fallback. Main calls only carry temperature when the user set
   `model.temperature` explicitly; no hook can delete it. Needs a core seam
   (per-model `fixed_temperature`/omission or a temperature recovery rung).
2. **One 400 round-trip per (route, model) per process** on aux lanes before
   core's rejection memory kicks in; the memory is in-process only.
3. **Range clamping is latent today**: core does not forward the caller's
   temperature to `build_api_kwargs_extras` (fixed kwarg sets at
   `chat_completions.py:664` and `auxiliary_client.py:6450`). The builder reads
   `context["temperature"]` defensively, so clamping activates unchanged if a
   future core forwards it; until then only pins and the reasoning gate act.
   Ranged-route callers within `[0, 2]` are already valid; out-of-range values
   on aux lanes are caught by core's strip rung.
