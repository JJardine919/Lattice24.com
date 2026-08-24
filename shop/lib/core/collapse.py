"""The single collapse entry point for this package.

Why this file exists
--------------------
`aoi_collapse.py` was found in 93 copies on this machine, byte-identical in all
but one. Every branch that reimplemented the call around it also reimplemented
the mistakes. This module is the only place the engine is called, and it makes
the three recurring mistakes impossible rather than merely discouraged:

  1. WRONG KEYS. The result keys are `chaos_level`, `normalized_chaos`,
     `control_norm`, `intent_magnitude`. Reading `result['chaos']` returns a
     silent 0.0 that is indistinguishable from a genuine null, and has already
     produced one false "the engine found nothing" report. `Reading` is a
     frozen dataclass, so a wrong name raises AttributeError.
  2. UNSTATED SCALE. Every output is scale-dependent: the same vector at
     ||v||=1, 10 and 50 gates 12, 13 and 0 of 24 dimensions. A number quoted
     without its norm is meaningless, so `Reading` carries `norm`.
  3. WRONG LENGTH. `aoi_collapse` takes exactly 24 numbers. Above ||v||~189 the
     octonion orthogonality assert fires and the engine crashes. Both are
     checked here, up front, with a message that says which rule was broken.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from aoi_collapse import aoi_collapse as _aoi_collapse  # noqa: E402

#: md5 of the canonical 24D engine. Ten+ locations on disk carry this hash.
CANONICAL_MD5 = "63b6bbb24e4451f0529c505fde50677b"

DIM = 24
#: ``octonion_shadow_decompose`` fails its 1e-8 orthogonality assert around
#: ||v||~189. Refuse well below that rather than crash inside the engine.
MAX_NORM = 100.0


def _vendored_md5() -> str:
    with open(os.path.join(_HERE, "aoi_collapse.py"), "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def verify_engine() -> str:
    """Raise if the vendored engine is not the canonical one."""
    got = _vendored_md5()
    if got != CANONICAL_MD5:
        raise RuntimeError(
            f"vendored aoi_collapse.py has md5 {got}, expected {CANONICAL_MD5}. "
            "This package pins ONE engine on purpose -- a swapped core silently "
            "changes every number downstream. Restore it or update CANONICAL_MD5 "
            "deliberately, with a note saying why."
        )
    return got


@dataclass(frozen=True)
class Reading:
    """One collapse reading, with the scale it was taken at.

    Frozen and explicitly named: there is no ``.chaos`` and no ``.get()``, so
    the silent-zero failure mode cannot recur.
    """

    chaos_level: float
    normalized_chaos: float
    intent_magnitude: float
    control_norm: float
    norm: float

    def as_row(self) -> tuple[float, ...]:
        return (self.chaos_level, self.normalized_chaos,
                self.intent_magnitude, self.control_norm, self.norm)


def prepare24(values: Sequence[float], *, centre: bool = True,
              norm: float = 1.0) -> np.ndarray:
    """Centre and rescale a 24-vector to an explicit working norm.

    Normalisation is mandatory before collapse and the target norm must be
    stated, because readings taken at different norms are not comparable. A
    constant input has no direction to scale and is returned centred at zero --
    that is the flat degenerate case, and the harness gate exists to catch it.
    """
    v = np.asarray(values, dtype=float)
    if v.shape != (DIM,):
        raise ValueError(
            f"aoi_collapse takes exactly {DIM} numbers, got shape {v.shape}. "
            "Downsample to 24 -- never switch engines to dodge the constraint."
        )
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite value in the 24-vector; clean it upstream")
    if centre:
        v = v - v.mean()
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros(DIM)          # flat degenerate; the gate will catch it
    return v * (norm / n)


def collapse24(values: Sequence[float]) -> Reading:
    """Collapse exactly 24 numbers. Returns a `Reading`, never a bare dict."""
    v = np.asarray(values, dtype=float)
    if v.shape != (DIM,):
        raise ValueError(f"aoi_collapse takes exactly {DIM} numbers, got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite value in the 24-vector")
    n = float(np.linalg.norm(v))
    if n > MAX_NORM:
        raise ValueError(
            f"||v|| = {n:.1f} exceeds {MAX_NORM}. The octonion decomposition "
            "loses orthogonality above ~189 and asserts. Normalise first, and "
            "state the norm you normalised to."
        )
    r = _aoi_collapse(v)
    missing = {"chaos_level", "normalized_chaos", "intent_magnitude",
               "control_norm"} - set(r)
    if missing:
        raise RuntimeError(f"engine returned no {sorted(missing)}; wrong engine?")
    return Reading(
        chaos_level=float(r["chaos_level"]),
        normalized_chaos=float(r["normalized_chaos"]),
        intent_magnitude=float(r["intent_magnitude"]),
        control_norm=float(r["control_norm"]),
        norm=n,
    )
