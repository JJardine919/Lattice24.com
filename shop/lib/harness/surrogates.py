"""Arm 5 (IAAFT surrogates) and Arm 6 (permutation entropy). Engine-free.

Why these two, and what each one kills
--------------------------------------
The four original arms share a weakness. C1 (own-shuffle) destroys the power
spectrum along with the ordering, so ANY smooth series beats it -- a ramp, a
decelerating aircraft, a drifting sensor. C3 (sorted surrogate) patches the
worst case by hand, one shape at a time. Neither is a matched null for
"smoothness in general".

ARM 5 -- IAAFT (Schreiber & Schmitz 1996, iteratively refined amplitude-
adjusted Fourier transform). A surrogate that matches the input in BOTH:
    * the value distribution, exactly (it is a permutation of the input), and
    * the power spectrum, to convergence.
Everything a linear Gaussian process can explain is therefore already in the
null. A sorted arrangement of the same numbers cannot score as a hit against
it, because the surrogate pool already carries that distribution and a
spectrum matched to the observed one. This is the arm that was missing when
grid, aviation and refinery each fired on shape rather than content.

    Exact-distribution matters here specifically. The vetted encoding is
    UNCENTRED, so the DC term is load-bearing (see harness.controls). An IAAFT
    surrogate is a permutation of the input, so mean, norm, variance and every
    higher moment are preserved to the bit. There is no encoding mismatch to
    argue about.

ARM 6 -- permutation entropy (Bandt & Pompe 2002), orders 3 and 4, normalised.
It counts ordinal patterns of successive points and nothing else, so it is
blind by construction to amplitude, DC offset and spread -- the three things
the uncentred collapse encoding actually responds to. If arm 6 fires where the
collapse is silent, the ordering carries information the engine did not read.

WHAT THE IAAFT NULL CANNOT DO -- read this before quoting an arm-5 null
----------------------------------------------------------------------
IAAFT preserves the autocorrelation function. That is its purpose. So an AR(1)
process -- pure linear persistence -- is INSIDE the IAAFT null by construction,
and no statistic scored against an IAAFT pool can fire on it. Measured, not
assumed: see `arm_validation.py`, case AR1.

That is not a defect to fix, it is the definition of the null, and it is
exactly why both arms are scored against TWO pools:

    vs IAAFT   -- "is there more here than the distribution and the spectrum?"
                  Answers the confound that killed grid/aviation/refinery.
    vs SHUFFLE -- "is there any ordering information at all?"
                  Catches linear persistence, which IAAFT deliberately keeps.

Quoting one without the other is how a screen ends up unable to fail.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: Surrogates per pool. 200 gives a two-sided permutation-p resolution of
#: 1/201 = 0.00498, so nothing below ~0.005 is expressible by this arm and a
#: Bonferroni-corrected claim over more than ~10 tests is out of its reach.
#: Say so rather than quoting p < 0.001 from a 200-draw pool.
N_SURROGATE = 200
#: IAAFT inner iterations. Convergence is detected by the rank order repeating,
#: which for n=24 normally happens well inside 50.
IAAFT_ITERS = 200


# --------------------------------------------------------------------------
# ARM 5 -- IAAFT surrogate
# --------------------------------------------------------------------------

def iaaft_surrogate(x: Sequence[float], rng: np.random.Generator, *,
                    n_iter: int = IAAFT_ITERS) -> np.ndarray:
    """One IAAFT surrogate: same values exactly, power spectrum matched.

    Alternates two projections until the rank order stops moving:
      (1) impose the target Fourier AMPLITUDES, keeping the current phases;
      (2) impose the target VALUE DISTRIBUTION by rank-remapping.
    Step (2) is applied last, so the returned series is always an exact
    permutation of the input -- mean, norm and variance are preserved to the
    bit and no encoding mismatch is introduced.
    """
    v = np.asarray(x, dtype=float)
    n = v.size
    sorted_v = np.sort(v)
    target_amp = np.abs(np.fft.rfft(v))

    y = rng.permutation(v)
    prev_ranks = None
    for _ in range(n_iter):
        phases = np.angle(np.fft.rfft(y))
        y = np.fft.irfft(target_amp * np.exp(1j * phases), n=n)
        ranks = np.argsort(np.argsort(y, kind="stable"), kind="stable")
        y = sorted_v[ranks]
        if prev_ranks is not None and np.array_equal(ranks, prev_ranks):
            break
        prev_ranks = ranks
    return y


def iaaft_pool(x: Sequence[float], rng: np.random.Generator, *,
               n: int = N_SURROGATE) -> list[np.ndarray]:
    return [iaaft_surrogate(x, rng) for _ in range(n)]


def spectrum_fidelity(x: Sequence[float], surrogates: Sequence[np.ndarray]) -> float:
    """Mean relative L2 error between target and surrogate power spectra.

    Reported with every arm-5 result. IAAFT trades a little spectral accuracy
    for an exact distribution; if that trade went badly the pool is not the
    null it claims to be, and this number is the only way to know.
    """
    a = np.abs(np.fft.rfft(np.asarray(x, float)))
    d = np.linalg.norm(a)
    if d < 1e-15 or not surrogates:
        return float("nan")
    return float(np.mean([np.linalg.norm(np.abs(np.fft.rfft(s)) - a) / d
                          for s in surrogates]))


# --------------------------------------------------------------------------
# ARM 6 -- permutation entropy
# --------------------------------------------------------------------------

def ordinal_patterns(x: Sequence[float], order: int, delay: int = 1) -> list[tuple]:
    """Bandt & Pompe ordinal patterns. Ties broken by index (stable sort)."""
    v = np.asarray(x, dtype=float)
    m = v.size - (order - 1) * delay
    if m < 1:
        return []
    return [tuple(np.argsort(v[i:i + (order - 1) * delay + 1:delay],
                             kind="stable"))
            for i in range(m)]


def perm_entropy(x: Sequence[float], order: int = 3, delay: int = 1,
                 normalise: bool = True) -> float:
    """Normalised permutation entropy in [0, 1]. 0 = one pattern only."""
    pats = ordinal_patterns(x, order, delay)
    if not pats:
        return float("nan")
    counts = np.array(list({p: pats.count(p) for p in set(pats)}.values()),
                      dtype=float)
    p = counts / counts.sum()
    h = float(-np.sum(p * np.log(p)))
    return h / math.log(math.factorial(order)) if normalise else h


def n_distinct_patterns(x: Sequence[float], order: int = 3,
                        delay: int = 1) -> int:
    return len(set(ordinal_patterns(x, order, delay)))


def pe_is_degenerate(x: Sequence[float], order: int = 3, delay: int = 1) -> bool:
    """True when the window realises ONE ordinal pattern and nothing else.

    A monotone series (a ramp, a sorted surrogate, a constant under stable tie
    breaking) has exactly one pattern at every order. Its permutation entropy
    is then 0 -- the minimum -- and it will sit far below any surrogate pool.
    Reading that as a hit would make arm 6 a ramp detector, which is the exact
    defect it was added to remove.

    This is a structural test, not a tuned threshold: it asks whether the
    statistic has more than one category to distribute over. With one category
    the entropy is zero for the same reason the flat-degenerate vector has
    chaos zero -- there is nothing to be entropic about. It is the same class
    of case as C0 and it is reported the same way: NOT RUN, never a pass.
    """
    return n_distinct_patterns(x, order, delay) <= 1
