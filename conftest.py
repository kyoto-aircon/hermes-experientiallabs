"""Pytest bootstrap for the standalone plugin repo.

This repo's root has an ``__init__.py`` (Hermes loads the plugin as a package),
so pytest builds a ``Package`` collector for the root and imports that
``__init__.py`` before running any test. That import needs Hermes' libraries on
``sys.path``, so we add the hermes-agent checkout here — at conftest import
time, which happens before the root package is imported.

The fixtures then load the plugin the same way Hermes' provider discovery does:
as a package with submodule search locations so the relative ``.account_usage``
import resolves.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
_PLUGIN_MODULE = "explabs_provider_plugin_under_test"

# Ensure the flat-import fallback in __init__.py resolves during collection.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _candidate_hermes_roots():
    env = os.environ.get("HERMES_AGENT_ROOT")
    if env:
        yield Path(env)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        yield Path(local) / "hermes" / "hermes-agent"
    home = Path.home()
    yield home / ".local" / "share" / "hermes" / "hermes-agent"
    yield home / "AppData" / "Local" / "hermes" / "hermes-agent"


def _venv_site_packages(hermes_root: Path):
    for venv in (hermes_root / "venv", hermes_root / ".venv"):
        if not venv.is_dir():
            continue
        yield from venv.glob("Lib/site-packages")
        yield from venv.glob("lib/python*/site-packages")


def _ensure_hermes_importable() -> bool:
    """Put Hermes (and any deps it needs) on ``sys.path``.

    Hermes' own pure-Python dependencies (e.g. ``ruamel.yaml`` via
    ``hermes_yaml``) may live in the Hermes venv rather than the interpreter
    running pytest. We append that venv's site-packages as a fallback so the
    checkout's modules import; appended, so the running interpreter's packages
    and stdlib always win, and version-tagged compiled wheels simply don't load.
    """
    hermes_root: Path | None = None
    try:
        import providers  # noqa: F401
    except ImportError:
        for candidate in _candidate_hermes_roots():
            if (candidate / "providers" / "__init__.py").is_file():
                hermes_root = candidate
                sys.path.insert(0, str(candidate))
                break
        else:
            return False

    if hermes_root is not None and importlib.util.find_spec("ruamel") is None:
        for site in _venv_site_packages(hermes_root):
            sys.path.append(str(site))
    return True


# Done at import time (not in a fixture) so the root package import succeeds.
_HERMES_AVAILABLE = _ensure_hermes_importable()


def _load_plugin():
    if _PLUGIN_MODULE in sys.modules:
        return sys.modules[_PLUGIN_MODULE]
    spec = importlib.util.spec_from_file_location(
        _PLUGIN_MODULE,
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PLUGIN_MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def plugin():
    if not _HERMES_AVAILABLE:
        pytest.skip(
            "hermes-agent not found; set HERMES_AGENT_ROOT to its checkout "
            "(e.g. C:\\Users\\<you>\\AppData\\Local\\hermes\\hermes-agent)"
        )
    return _load_plugin()


@pytest.fixture(scope="session")
def profile(plugin):
    import providers

    resolved = providers.get_provider_profile("experientiallabs")
    assert resolved is not None, "experientiallabs profile must be registered"
    return resolved


@pytest.fixture(scope="session")
def account_usage(plugin):
    import importlib

    return importlib.import_module(f"{_PLUGIN_MODULE}.account_usage")
