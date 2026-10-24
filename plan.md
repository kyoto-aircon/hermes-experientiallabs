This plan was AI-generated with minimal to no oversight.

# Plan: slim the plugin settings + classify gateway guardrail 400s

## 1. Scope contract

**Goal** (user request, verbatim):
1. "are plugging settings actually useful? or can we disperse those settings throughout the native hermes ui, because i can set an api key in the plugin settings and in the 'providers -> api keys' tab in the settings"
2. "when chatting to exp labs, i keep getting error HTTP 400: Content inspection does not support this complete request. Use a supported context or modality. and content_policy_blocked: HTTP 400: A gateway guardrail could not complete this request. logs are in C:\Users\Spheraz\AppData\Local\hermes\profiles\wraith\logs"

**In scope**
- Q1: remove the redundant `api_key` and `base_url` secret fields from `plugin.yaml` `config_schema` (Hermes' native Settings → Providers → "API keys" tab already edits both, including the `EXPLABS_BASE_URL` advanced override). Keep `cost_api_base` + `usage_timeout_s` (no native home — they must stay in the plugin settings page).
- Q2: add a `classify_api_error` hook to the provider profile that marks the two exact gateway guardrail 400s as **retryable**, so Hermes retries with backoff instead of killing the turn. Root cause is gateway-side; the plugin can only mitigate.
- Bump versions (plugin.yaml + pyproject.toml) 0.3.0 → 0.4.0; update tests; update AGENTS.md and README docs.

**Out of scope** (with why)
- Fixing the Experiential Labs gateway guardrail service — it's their hosted platform; not our code.
- Any Hermes core change — Hermes already provides the `classify_api_error` hook we need.
- Retry-on-fallback / credential-rotation behavior beyond marking these two errors retryable.
- Removing/merging `cost_api_base`/`usage_timeout_s` — they have no native Hermes UI surface.

**Assumptions**
- Removing the schema fields does not delete existing `EXPLABS_API_KEY`/`EXPLABS_BASE_URL` entries from the profile `.env`; runtime reads `env_vars` directly, so existing keys keep working.
- The guardrail failures are transient (evidence: identical requests succeed in other windows); retries have a good chance of passing. If a failure is deterministic, Hermes' retry budget caps the cost (400s fail fast).

**Definition of done** (testable)
- `plugin.yaml` `config_schema` contains only `cost_api_base` and `usage_timeout_s`.
- `python -m pytest` passes with updated tests (offline suite).
- New unit tests prove the classifier hook returns `{"reason": "content_policy_blocked", "retryable": True}` for both exact gateway messages on status 400, and `None` for anything else.
- Versions bumped in sync; `test_manifest.py` green.
- AGENTS.md "Known gateway quirks" documents both 400s and the plugin's retry classification.

**Size**: S (4 files edited + 1 test file created; no new deps; no data/schema changes).

**Open questions**
- None blocking. Minor: exact Hermes version that introduced the `ProviderProfile.classify_api_error` field — if unknown, keep `requires_hermes` at the installed Hermes version used for dev (verifiable with `hermes plugins doctor .`). Recommend reporting the gateway outage to Experiential Labs via their Discord (discord.gg/B6sM8xTVwU) — user action, not code.

## 2. Findings

**Q1 — settings redundancy (explored in Hermes checkout)**
- Desktop Settings → Providers → "API keys" (`providers-settings.tsx`) supports plugin-registered providers: `hermes_cli/provider_catalog.py` + `sync_plugin_provider_catalog()` + `auth_plugin_providers.py` mirror every registered plugin profile's `env_vars` into the Keys tab. Writes go to the profile `.env` via `PUT /api/env` → `save_env_value` (same destination as the plugin's secret fields).
- The `*_BASE_URL` entry of `env_vars` is auto-rendered as an advanced "base URL override" field on the same card (`config_env.py:239-244`); no config_schema needed.
- First-party provider plugins (anthropic, openrouter, custom) ship **no** `config_schema` at all — credentials are declared purely via `env_vars` in `__init__.py`.
- The two plugin secret fields and the native tab are unsynced duplicate editors of the same `.env` lines (last write wins). TUI plugins hub doesn't render config_schema anyway.
- Verdict: `api_key` + `base_url` are redundant → remove. `cost_api_base` + `usage_timeout_s` have no native home → keep.

**Q2 — gateway 400s (log forensics, agent.log)**
- Error A: code `unsupported_capability`, "Content inspection does not support this complete request. Use a supported context or modality."
- Error B: code `content_filter`, "A gateway guardrail could not complete this request."
- Same request shapes pass or fail purely time-correlated (32 consecutive successes 19:01–19:09:30, mass failures 18:26–18:42 and 19:09:31+), including tool-result payloads and 50k-token contexts that worked earlier. No images in any request. → The gateway's content-inspection/guardrail service fails intermittently; not a Hermes or plugin bug.
- Hermes classifies both as non-retryable today (`content_policy_blocked`), so turns die on first hit.
- Hermes escape hatch: `ProviderProfile.classify_api_error` field (consumed at `agent/error_classifier.py:758-798`, stage 2 of `_STAGES`, before built-in status handling). Returns `{"reason", "retryable", ...}`; `agent/turn_api_error.py` consumes `classified.retryable` directly — a `retryable: True` hint makes Hermes retry with backoff.

## 3. Steps

1. **plugin.yaml** — remove `api_key` and `base_url` from `config_schema` (keep `cost_api_base`, `usage_timeout_s`); bump `version` to 0.4.0.
2. **pyproject.toml** — bump `[project]` `version` to 0.4.0 (version sync rule).
3. **__init__.py** — add module-level pure function `classify_experientiallabs_api_error(error, *, status_code, error_code, message, body, model)`: return `None` unless `status_code == 400` and `error_code` (falling back to `body.error.code` when `error_code` is empty) is in `{"unsupported_capability", "content_filter"}` and `message` contains one of the two gateway service-failure phrases (case-insensitive, trailing punctuation not required: "Content inspection does not support this complete request" / "A gateway guardrail could not complete this request"); match → `{"reason": "content_policy_blocked", "retryable": True}`. Defensive (never raise). Pass it as `classify_api_error=...` in the `ExperientialLabsProfile` constructor.
4. **tests/test_classify.py (new)** — unit tests for the pure classifier: both messages × both codes → retryable verdict; `error_code` empty with matching `body.error.code` → retryable; other 400s / other statuses / non-matching text → `None`; non-dict body → no crash.
5. **tests/test_manifest.py** — update `config_schema` key set to `{cost_api_base, usage_timeout_s}`; drop `test_secret_fields_write_the_profile_env_vars` (no secret fields remain); keep defaults/documentation/type-warning tests for the two remaining fields; update version assert to 0.4.0.
6. **AGENTS.md** — rewrite the "Optional settings" rule: keys are set via Hermes' native Settings → Providers → API keys (or CLI wizard / `/model` picker in the TUI); only `cost_api_base`/`usage_timeout_s` remain in `config_schema`. Add a "Known gateway quirks" entry documenting both guardrail 400s, that the plugin classifies them retryable, and the Discord link for reporting outages.
7. **README.md** — align the Settings section (if it documents the removed fields) with the native Keys tab + the two remaining knobs; mention the guardrail quirk.
8. **Verify** — `python -m pytest`; if the Hermes CLI is available, `hermes plugins doctor .` for manifest + profile sanity. Check that the installed Hermes version has the `classify_api_error` profile field (it does, per the dev checkout); if the feature's introduction version is unknown, note it in a comment rather than raising `requires_hermes` blindly.

## 4. Files

- Edit: `plugin.yaml`, `pyproject.toml`, `__init__.py`, `tests/test_manifest.py`, `AGENTS.md`, `README.md`
- Create: `tests/test_classify.py`
- Delete: none

## 5. Risks

- **Older Hermes without `classify_api_error`**: dataclass constructor would raise on the unknown kwarg → plugin fails to load. Mitigate: the field exists in the dev checkout; keep `requires_hermes` floor sane and confirm with `hermes plugins doctor .`. If the introduction version can't be pinned, document it.
- **Retrying genuinely blocked content**: retries are scoped to the two exact gateway-service-failure messages, not real content blocks; 400s fail fast so worst case is a few wasted cheap requests within Hermes' retry budget.
- **Removed secret fields vs existing installs**: existing `.env` entries persist and continue to be read via `env_vars`; the native Keys tab becomes the single editor. Verify by checking `resolve_usage_settings` never touches the removed fields (it only reads the two non-secret ones).
- **Version drift**: `test_manifest.py` cross-checks plugin.yaml ↔ pyproject.toml versions and defaults ↔ module constants; keep them in sync in one commit.
