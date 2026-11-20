# Plan: v0.8.0 — close the remaining Nous-parity gaps

Six parallel workstreams, one git worktree + branch each, merged by the
parent agent in the order below. Ecosystem items are recorded as future
ideas, not in scope here.

## Ground rules (every workstream)

- Work ONLY in the assigned worktree. Never edit
  `~/.hermes/plugins/hermes-experientiallabs` (the live install), never
  change `~/.hermes` config, never restart the gateway, never push.
- Follow `AGENTS.md`: fail open, money is `Decimal`, pure builders, lazy
  imports (stdlib only at module load), offline tests by default
  (`integration`/`live` markers deselected).
- Do NOT bump `plugin.yaml` / `pyproject.toml` versions or the version
  assert in `tests/test_manifest.py` — the parent bumps to 0.8.0 once at
  merge time.
- Post-v0.21.6 `ProviderProfile` fields go through the existing
  `_OPTIONAL_PROFILE_FIELDS` guard in `__init__.py` (CI tests against the
  v0.21.6 floor).
- Prefer new modules over growing existing ones; keep `__init__.py` edits
  surgical (distinct regions per workstream) to ease merges.
- Tests: `HERMES_AGENT_ROOT=/home/vesta/.hermes/hermes-agent /usr/bin/python3 -m pytest -q`
  from the worktree root. All existing tests must stay green.
- Commits: lowercase imperative, `git commit --no-gpg-sign` (global
  `commit.gpgsign=true` but no secret key exists for the noreply identity).
- Never print API keys. `EXPLABS_API_KEY` lives in `~/.hermes/.env`; read
  it in-process only. Live paid completions: avoid unless the workstream
  explicitly budgets them; keyless catalog reads are free.
- Hermes core source for reading contracts:
  `/home/vesta/.hermes/hermes-agent` (providers/base.py, agent/*,
  hermes_cli/*).

## W1 — sampling clamp (feat/sampling-clamp)

Items: temperature pins + `sampling_requires_reasoning_none`.

Live facts: 9 routes pin `temperature` (`minimum_temperature ==
maximum_temperature == 1.0`, Claude 5 family); 73 routes have
`supports_temperature: false`. Aux title calls send a temperature and 400
on those routes today.

- Parse `supports_temperature`, `minimum_temperature`, `maximum_temperature`,
  `sampling_requires_reasoning_none` into catalog entries (`catalog.py`).
- In `build_api_kwargs_extras`, when the cached entry pins temperature,
  set top-level `temperature` to the pinned value; when a range is
  published, clamp the caller's value into it (top-level kwargs override
  `_base_kwargs`' temperature — verified in
  `agent/transports/chat_completions.py::_build_kwargs_from_profile`).
- For `supports_temperature: false` routes a hook cannot DELETE the field;
  research the classifier/field-rejection ladder
  (`agent/error_classifier.py`, `agent/turn_api_error.py`) for a safe
  strip-and-retry verdict on 400 `unsupported_parameter`/`invalid_parameter`
  naming `temperature`. If no safe mechanism exists, ship the clamp only
  and document the residual limitation — do not invent a retry loop.
- DoD: unit tests prove pinned routes get `temperature=1.0`, ranged routes
  clamp, cold cache leaves the caller's value untouched; full suite green.

## W2 — reasoning carriers (feat/reasoning-carriers)

Item: preserved-thinking lanes lose thinking on every tool turn because
core's DeepSeek echo rules match provider `deepseek`, not `experientiallabs`.

Live facts: `reasoning_content_native: true` on `deepseek-r1-0528`,
`deepseek-v4.1-flash(-uncensored)`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`,
`glm-5.1`, `glm-5.3-flash-uncensored`, `glm-5.3-uncensored`, `hy3`, `hy4`, …
Gateway docs: on a tool turn of a preserved-thinking lane the
`reasoning_content` is an opaque carrier; replaying it verbatim keeps
thinking; stripping is always safe; edited turns drop it with disclosure,
never a 400.

- Parse `reasoning_content_native` (and `reasoning_output_exposed`) into
  catalog entries.
- Read `agent/reasoning_carriers.py` + `agent/message_sanitization.py` for
  the capture/replay contract, then declare `message_carrier_text`
  (post-floor field → `_OPTIONAL_PROFILE_FIELDS` guard) and implement
  `replay_message_carriers` gated per model on `reasoning_content_native`
  so the carrier rides only on lanes that read it.
- DoD: offline tests prove capture/replay shaping for a native lane and
  omission for a non-native lane (drive `shape_wire_carriers` directly if
  practical); full suite green.

## W3 — retry safety (feat/retry-safety)

Items: Idempotency-Key replay safety + honest 429 reset hints.

- 429 resets: the classify hook receives the raw `error`; OpenAI SDK
  status errors carry `.response.headers`. Extract `Retry-After` /
  rate-limit reset headers defensively and return
  `error_context={"reset_at": ...}` in the verdict — first verify what
  `agent/turn_api_error.py` / `agent/rate_limit_tracker.py` actually
  consume (Nous' guard reads `error_context["reset_at"]`).
- Idempotency-Key: gateway honors it on chat/responses; same key + same
  body replays, same key + different body 409s, and a fresh key can
  double-bill. Profile hooks (`request_headers` is memoized per argument
  set; `build_api_kwargs_extras` gets no message content) may not offer a
  stable per-operation seam. Research `hermes_cli/middleware.py`
  (`apply_llm_request_middleware`, which sees `task_id`/`turn_id`/
  `api_request_id`) and `agent/turn_api_request.py` for a safe injection
  point. Ship ONLY if a key can be stable across retries of one logical
  request and unique across operations; otherwise deliver findings and a
  default-off design, not a double-bill risk.
- DoD: 429 verdicts carry reset hints when headers exist (unit tests with
  fake SDK errors); idempotency either proven-safe implementation or a
  written findings section in the branch's `plan-w3.md`; full suite green.

## W4 — picker UX (feat/picker-ux)

Items: `:fast` variants, cache-first `picker_catalog`, static row/placement.

- Merge `<slug>:fast` into `fetch_models` for models the catalog attests a
  fast path (`service_tier_pricing_enabled` or `pricing.priority`; live
  today: `deepseek-v4.1-flash-uncensored`, `gpt-5.6-sol`). Derive from the
  cached catalog, never hardcode; cold cache adds nothing (existing
  `test_free_models.py` exact-list assertions must stay green).
- Add a `picker_catalog(force_refresh)` hook (post-floor field, guarded):
  cache-first list from `catalog.get_cached_entries` + promo/free/fast
  merge; `force_refresh=True` does the live fetch. Contract: returns
  `list[str] | None` (see `hermes_cli/models.py::_catalog_fetcher`).
- Set `static_models=_FALLBACK_MODELS` and deliberate placement
  (`picker_after`/`registry_after`, e.g. after `openrouter`) via the
  guarded optional-fields dict.
- DoD: new offline tests for fast merge + picker hook; full suite green.

## W5 — Responses wire opt-in (feat/responses-wire)

Item: native OpenAI Responses route (`/v1/responses`) with encrypted
reasoning replay, default OFF.

- Add a non-secret `config_schema` knob (e.g. `responses_wire: bool,
  default false`) in `plugin.yaml` + update `tests/test_manifest.py`
  key-set/default assertions in this branch (this workstream owns the
  manifest schema change).
- Implement `api_mode_for_model` returning `"codex_responses"` for
  OpenAI-family slugs only when the knob is on; research
  `agent/transports/codex.py` + the `responses_*` profile fields for what
  the gateway needs (encrypted reasoning include/replay, `store`,
  `previous_response_id` — gateway keeps continuations 30 days only with
  capture on; ZDR requests 400 `zdr_continuation_disabled`).
- Live verification budget: ≤15 tiny completions on `gpt-6-luna` or
  `deepseek-v4-flash`, `live`-marked tests only. Measure whether reasoning
  replay round-trips; do NOT enable by default regardless of result.
- DoD: default-off behavior proven by offline tests (chat wire unchanged);
  live findings recorded in `plan-w5.md`; full suite green.

## W6 — media backends (feat/media-backends)

Item: one key beyond chat. New self-contained sibling plugin dirs under
`media/` in the worktree; zero edits to the core provider files.

- Image generation: `media/image-gen-experientiallabs/` with
  `kind: backend`, mirroring
  `/home/vesta/.hermes/hermes-agent/plugins/image_gen/openrouter/`
  (registration API, manifest, tests). Endpoint
  `POST https://api.experientiallabs.ai/v1/images/generations` (+ edits),
  key `EXPLABS_API_KEY`, models whose catalog modalities include image
  output. Offline tests with stubbed HTTP; `hermes plugins doctor <dir>`
  must pass.
- TTS/STT: research the seams (`tools/tts_tool_openai.py`,
  `tools/transcription_tools.py`, `hermes_cli/tools_config_providers.py`).
  If a plugin seam exists, build `media/tts-experientiallabs/` and/or
  `media/stt-experientiallabs/` the same way; if config-only, write the
  exact working config in `media/README.md` instead of forcing a plugin.
- DoD: image backend loads + offline tests green; TTS/STT either working
  plugins or documented config; findings in `media/README.md`.

## Merge order (parent)

1. W1 sampling → 2. W2 carriers → 3. W3 retry-safety → 4. W4 picker →
5. W5 responses → 6. W6 media (disjoint dirs, anytime).
After each merge: full suite + `hermes plugins doctor`. Conflicts expected
in `__init__.py` (constructor/guard dict) and `catalog.py` parse functions —
resolve by keeping both sides' distinct regions. Final: bump 0.8.0
(plugin.yaml + pyproject.toml + manifest test), README/AGENTS refresh,
one release commit, push, CI watch.

## Future ideas (ecosystem — NOT in this plan)

- Submit to the official Hermes plugin catalog (community tier) + tag
  releases; README's `hermes plugins install hermes-experientiallabs`
  line is aspirational until then.
- Contribute an Experiential Labs provider to `rarf/hermes-quota-plugin`
  (its `docs/add-provider.md`) for the desktop quota widget.
- Nightly live CI workflow with a repo-secret key to catch gateway schema
  drift (integration tests exist but only run manually).
- Embeddings/video backends if Hermes grows subsystem seams for them.
- `:online` web-search variants in the picker (Hermes has its own web
  tools; low value today).
