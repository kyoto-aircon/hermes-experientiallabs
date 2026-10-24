# Experiential Labs provider for Hermes Agent

[![tests](https://github.com/kyoto-aircon/hermes-experientiallabs/actions/workflows/ci.yml/badge.svg)](https://github.com/kyoto-aircon/hermes-experientiallabs/actions/workflows/ci.yml)

Adds [experientiallabs.ai](https://experientiallabs.ai) — an OpenAI-compatible
multi-model gateway (~900 models) — as a first-class provider in
[Hermes Agent](https://hermes-agent.nousresearch.com).

The plugin is a standard Hermes **model-provider** plugin. Registering the
profile automatically wires it into:

- the auth registry and setup wizard
- Desktop **Settings → Providers** (keys tab) and `hermes model`
- the `/model` picker (live `GET /v1/models`, with a curated offline fallback)
- `hermes doctor` (a `/models` health check)
- `/usage` (wallet balance + daily key cap)

## Install

```bash
hermes plugins install hermes-experientiallabs   # official Hermes plugin catalog (community tier)
# or install straight from the repo:
hermes plugins install https://github.com/kyoto-aircon/hermes-experientiallabs
```

Or drop this directory into `$HERMES_HOME/plugins/hermes-experientiallabs/`
(the manifest declares `kind: model-provider`, so provider discovery finds it).

Then add your key:

```bash
hermes auth add experientiallabs
# or put it in ~/.hermes/.env:
# EXPLABS_API_KEY=xpl_...
```

Get a key at <https://platform.experientiallabs.ai/settings/api-keys>.

## Usage

```bash
hermes model                       # pick Experiential Labs, then a model slug
hermes -z -m experientiallabs:qwen3.8-27b "hello"
```

`/model` accepts the slug directly, e.g. `experientiallabs:deepseek-v4-flash`.

## Configuration

Everything is optional — the defaults work against the public gateway, and an
unconfigured plugin is simply a provider without a key.

**Credentials** are not plugin settings. Set the key in Desktop → **Settings →
Providers → API keys** (the plugin's provider card appears there natively), or
with `hermes auth add experientiallabs`, the setup wizard, or `/model` in the
TUI. That card also carries the advanced **base URL override**
(`EXPLABS_BASE_URL`) — nothing is duplicated in the plugin's own form.

**The two `/usage` knobs** live in Desktop → **Settings → Plugins →
Experiential Labs**. The TUI plugins hub does not render plugin forms; set
these in `config.yaml` there.

| Setting | Kind | Default | What it does |
| --- | --- | --- | --- |
| **Cost API base URL** | string | `https://api.experientiallabs.ai/api/v1` | **Advanced** — base for the `/usage` endpoints. Only set this if the `EXPLABS_BASE_URL` chat mirror also serves the Cost API (`GET /credits`, `GET /key`). |
| **Usage fetch timeout (seconds)** | float | `4.0` | **Advanced** — per-request timeout for the two `/usage` GETs. Hermes allows the usage hook 10s total, so values above `4.5` are clamped. |

These live under `plugins.entries.hermes-experientiallabs.settings` in
`config.yaml` if you prefer editing files directly. The key and base URL are
read from `.env` at process start (so they apply after a Hermes restart); the
two knobs are read on each `/usage` call.

### Profile defaults (not settings)

These are the provider's contract with Hermes, not user configuration —
they stay pinned in the profile like every bundled provider:

| Default | Value |
| --- | --- |
| Base URL | `https://api.experientiallabs.ai/v1` |
| API key env vars | `EXPLABS_API_KEY`, then `EXPERIENTIALLABS_API_KEY` |
| API mode | `chat_completions` |
| Auxiliary model | `glm-5.3-flash` (cheap, vision-capable) |
| Curated fallbacks | `qwen3.8-27b`, `deepseek-v4-flash`, `gpt-6-luna`, `claude-haiku-4.5`, `gpt-6-astra` |

`fallback_models` is the curated model-picker floor Hermes shows when live
model listing fails, and `default_aux_model` feeds Hermes' auxiliary-model
ladder — Hermes consumes both from the profile by design.

### Account usage

`/usage` calls `GET /api/v1/credits` for the wallet balance and
`GET /api/v1/key` for the key's daily USD cap (resets 00:00 UTC). Both are
best-effort: the hook returns nothing rather than failing. The two advanced
settings above (Cost API base URL, timeout) apply to these calls only.

## Privacy / network

The plugin only talks to Experiential Labs hosts (aside from any base URL you
configure), and only when Hermes invokes it:

- `https://api.experientiallabs.ai/v1` — chat completions and the model list
  (via Hermes' own transport, `/model` picker and doctor health check)
- `https://api.experientiallabs.ai/api/v1` — `GET /credits` and `GET /key`,
  only when the `/usage` view asks for a snapshot

No telemetry, no other endpoints. Credentials are resolved through Hermes'
own routes (`hermes auth`, `.env`) — the plugin never reads or stores keys
itself.

## Notes / limitations

- Some routes pin sampling parameters (the Claude 5 family pins
  `temperature`). Hermes' session-title side call sends a temperature, so those
  models can 400 on auxiliary calls and fall back to the default. This is
  gateway behaviour, not a plugin setting.
- `prompt_cache_key` is deliberately not advertised (`supports_prompt_cache_key`
  is left off); the gateway treats it as an affinity hint, so there is nothing
  to gain and it avoids sending an unknown top-level field.
- The gateway's content-inspection/guardrail service intermittently fails
  closed with HTTP 400 (codes `unsupported_capability` and `content_filter`)
  before a request reaches a model; identical requests succeed in other
  windows. The profile classifies exactly those two signatures as retryable,
  so Hermes backs off and retries instead of ending the turn. Real content
  refusals keep the built-in non-retryable verdict. If an outage persists,
  report it to Experiential Labs ([Discord](https://discord.gg/B6sM8xTVwU)).

## Development

```bash
python -m pytest
hermes plugins doctor .
```

The tests import the real Hermes `providers` package, so they auto-detect a
standard `hermes-agent` install (and bridge its venv's pure-Python deps when
pytest runs under a different interpreter). Set `HERMES_AGENT_ROOT` if your
checkout lives elsewhere.

CI runs the same suite on every push and pull request by shallow-cloning the
`v0.21.6` release of `NousResearch/hermes-agent` (the floor `requires_hermes`
declares), syncing its locked dependencies with uv, and running pytest against
it (see `.github/workflows/ci.yml`).

## AI-generated

This repository was written primarily by AI coding agents under the owner's
direction — see [`AI_DISCLAIMER.md`](AI_DISCLAIMER.md).

MIT licensed.
