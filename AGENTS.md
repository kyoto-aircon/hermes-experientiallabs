# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A Hermes Agent **model-provider plugin** (`kind: model-provider`) that
registers Experiential Labs (experientiallabs.ai, an OpenAI-compatible
gateway) as a first-class provider. Hermes loads it via plugin discovery
(`plugin.yaml` + `__init__.py`) — it is never installed as a Python package.

## Layout

- `plugin.yaml` — manifest: `name: experientiallabs-provider`, `kind:
  model-provider`, `version` (bump for releases).
- `__init__.py` — the provider profile (`ExperientialLabsProfile`): env vars,
  base URL, fallback models, aux model, and the `fetch_account_usage` hook.
- `account_usage.py` — the `/usage` snapshot: wallet balance from
  `GET /api/v1/credits` and the key's daily USD cap from `GET /api/v1/key`
  (resets 00:00 UTC).
- `conftest.py` — pytest bootstrap: finds a real `hermes-agent` checkout,
  bridges its venv deps, and loads the plugin as a package the way Hermes'
  provider discovery does.
- `tests/` — `test_profile.py`, `test_account_usage.py`.
- `pyproject.toml` — pytest config plus a minimal `[project]` table. There is
  intentionally no build backend; the `[project]` version only exists because
  Hermes' plugin manager copies this file into a uv workspace and `uv lock`
  rejects a version-less `[project]`.

## Commands

```bash
python -m pytest            # offline suite (integration/live deselected)
python -m pytest -m live    # live gateway tests (needs EXPLABS_API_KEY)
hermes plugins doctor .     # manifest + profile sanity check (needs Hermes)
```

Hermes checkout discovery for tests: `HERMES_AGENT_ROOT`, then
`%LOCALAPPDATA%\hermes\hermes-agent`, `~/.local/share/hermes/hermes-agent`,
`~/AppData/Local/hermes/hermes-agent`.

## Rules

- **Fail open.** `fetch_account_usage` must never raise into Hermes; it
  returns `None` on any failure, exactly like the dispatch boundary expects.
- **Money is `Decimal`, never `float`** — the Cost API returns decimal
  strings. Keep present-but-zero distinct from absent (`_pick` exists for
  that; a hard `$0` cap is not "no cap").
- **Pure builders.** The response→window builders in `account_usage.py` do no
  I/O and stay unit-testable; only the fetch path talks to the network.
- **Lazy imports.** Only stdlib at module load; `httpx` and Hermes internals
  import inside the fetch path.
- **Budget.** Two sequential GETs with 4s timeouts each — Hermes' hook
  deadline is 10s.
- **Tests are offline by default.** Anything touching the real gateway gets
  `@pytest.mark.integration` or `@pytest.mark.live` (both deselected by
  `addopts`).
- Env vars, in priority order: `EXPLABS_API_KEY` (primary),
  `EXPERIENTIALLABS_API_KEY` (fallback), `EXPLABS_BASE_URL` (base-URL
  override; the trailing `*_BASE_URL` entry in the profile's `env_vars` is
  what makes Hermes honour it).
- API surface: serving `https://api.experientiallabs.ai/v1` (Chat
  Completions, `GET /v1/models`); Cost API
  `https://api.experientiallabs.ai/api/v1` (`/credits`, `/key`); public
  catalog `GET /api/models?limit=1000` (keyless; default limit is 500).
- Commits: short, lowercase, imperative ("add x", "fix y").

## Known gateway quirks (do not "fix" these in the plugin)

- Some routes pin sampling parameters (the Claude 5 family pins
  `temperature`), so auxiliary calls that send a temperature can 400 —
  gateway behaviour, not a plugin setting.
- `prompt_cache_key` is deliberately not advertised: the gateway treats it as
  an affinity hint, so there is nothing to gain and it avoids sending an
  unknown top-level field.

## Environment

Windows + PowerShell here. Use `python -m pytest` (not bare `pytest`) so the
right interpreter resolves. Never print API keys; read `EXPLABS_API_KEY` from
the environment in-process only.

## Related

`kyoto-aircon/opencode-experientiallabs` integrates the same gateway into
OpenCode (TypeScript); the gateway API shapes and quirks documented there
apply here too. `rarf/hermes-quota-plugin` has a quota-widget provider for
Experiential Labs (see its `docs/add-provider.md`).
