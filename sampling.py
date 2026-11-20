"""Per-model sampling (temperature) constraints from the Experiential Labs catalog.

Pure builders: no I/O and only stdlib at module load, so they stay unit-testable
offline. ``ExperientialLabsProfile.build_api_kwargs_extras`` reads the cached
``/v1/models`` entry for the request's model and asks
``resolve_temperature_override`` which top-level ``temperature`` to force.

Why override instead of strip: Hermes applies a profile's top-level kwargs
AFTER ``_base_kwargs`` (``agent/transports/chat_completions.py``), so an
override wins over the caller's temperature — but no hook can DELETE a field
the caller already set. Routes with ``supports_temperature: false`` therefore
leave the caller's value untouched; core's auxiliary ladder already strips and
remembers temperature rejections there (see ``plan-w1.md`` for the full
findings and the residual main-path limitation).
"""

from __future__ import annotations

import math
from typing import Any


def _finite_float(value: Any) -> float | None:
    """Parse *value* into a finite ``float`` (``bool``/NaN/inf rejected), else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def entry_temperature_bounds(
    entry: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    """``(minimum, maximum)`` temperature bounds attested by a catalog entry.

    Re-validated defensively: entries also arrive from the disk mirror, which
    may predate a schema change or hold hand-edited values.
    """
    if not isinstance(entry, dict):
        return None, None
    return (
        _finite_float(entry.get("minimum_temperature")),
        _finite_float(entry.get("maximum_temperature")),
    )


def resolve_temperature_override(
    entry: dict[str, Any] | None,
    caller_temperature: Any = None,
    *,
    reasoning_disabled: bool = False,
) -> float | None:
    """Top-level ``temperature`` to force for *entry*, or ``None`` to leave the
    request untouched.

    - Absent/cold entry -> ``None`` (fail open; never guess constraints).
    - ``supports_temperature is False`` -> ``None``: the route rejects the
      field outright and a hook cannot delete the caller's value.
    - ``sampling_requires_reasoning_none is True`` and reasoning is not known
      to be disabled -> ``None``: sampling is only accepted with reasoning off
      (live GPT-5.x/6.x lanes), so adding or clamping a temperature while
      reasoning may be active would itself 400.
    - Pinned route (``minimum == maximum``) -> the pinned value, whatever the
      caller asked for (live Claude 5 family pins ``1.0``).
    - Published range -> the caller's temperature clamped into it, but ONLY
      when the caller's value is known. Hermes does not currently forward it
      to ``build_api_kwargs_extras``; unknown -> ``None``, never a guess.
    - Contradictory bounds (``min > max``) or no bounds -> ``None`` (fail open).
    """
    if not isinstance(entry, dict):
        return None
    if entry.get("supports_temperature") is False:
        return None
    if entry.get("sampling_requires_reasoning_none") is True and not reasoning_disabled:
        return None
    lo, hi = entry_temperature_bounds(entry)
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None:
        if lo > hi:
            return None
        if lo == hi:
            return lo
    caller = _finite_float(caller_temperature)
    if caller is None:
        return None
    if lo is not None and caller < lo:
        return lo
    if hi is not None and caller > hi:
        return hi
    return caller
