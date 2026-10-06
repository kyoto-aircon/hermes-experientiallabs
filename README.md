# Experiential Labs provider for Hermes Agent

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
hermes plugins install https://github.com/epicawesomesauce/explabs-provider-plugin
```

Or drop this directory into `$HERMES_HOME/plugins/experientiallabs-provider/`
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

| Setting | Default |
| --- | --- |
| Base URL | `https://api.experientiallabs.ai/v1` |
| API key env vars | `EXPLABS_API_KEY`, then `EXPERIENTIALLABS_API_KEY` |
| API mode | `chat_completions` |
| Auxiliary model | `glm-5.3-flash` (cheap, vision-capable) |
| Curated fallbacks | `qwen3.8-27b`, `deepseek-v4-flash`, `gpt-6-luna`, `claude-haiku-4.5`, `gpt-6-astra` |

Override the base URL with `EXPLABS_BASE_URL` (declared in the profile's
`env_vars`, so Hermes honours it) or by pinning `model.provider: experientiallabs`
together with `model.base_url` in `config.yaml`.

### Account usage

`/usage` calls `GET /api/v1/credits` for the wallet balance and
`GET /api/v1/key` for the key's daily USD cap (resets 00:00 UTC). Both are
best-effort: the hook returns nothing rather than failing.

## Notes / limitations

- Some routes pin sampling parameters (the Claude 5 family pins
  `temperature`). Hermes' session-title side call sends a temperature, so those
  models can 400 on auxiliary calls and fall back to the default. This is
  gateway behaviour, not a plugin setting.
- `prompt_cache_key` is deliberately not advertised (`supports_prompt_cache_key`
  is left off); the gateway treats it as an affinity hint, so there is nothing
  to gain and it avoids sending an unknown top-level field.

## Development

```bash
python -m pytest
hermes plugins doctor .
```

The tests import the real Hermes `providers` package, so they auto-detect a
standard `hermes-agent` install (and bridge its venv's pure-Python deps when
pytest runs under a different interpreter). Set `HERMES_AGENT_ROOT` if your
checkout lives elsewhere.

MIT licensed.
