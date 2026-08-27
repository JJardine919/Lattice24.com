"""Does the ORDER of a channel's rows carry information? -- the customer-facing arm.

WHY THIS FILE EXISTS
--------------------
It replaces `analyze.perm_order`, which was measured backwards on 2026-08-26
(`~/BUILDS/SHOP_TEST_2026-08-26.md`). That test returned the UPPER tail of an
RMS-successive-difference statistic against a plain shuffle. Real ordered data
is smoother than its own shuffle, so the observed value sits in the LOWER tail
and the printed p went to 1.0 -- which the report then described as "no sequence
structure". A pure ramp scored p = 1.0000 and was reported as structureless; the
same file with all rows scrambled scored p = 0.0250 and read as MORE structured
than the correct one.

Flipping the tail was tested and rejected: flipped, a straight line scores
p = 0.0010 and reads as SEQUENCE STRUCTURE. That trades a test that cannot fire
for one that cannot stay silent. The statistic needed a comparator a line fails.

WHAT REPLACED IT
----------------
Permutation entropy (Bandt & Pompe 2002, order 3) scored against a shuffle pool
of THE SAME 24 VALUES. Two properties make it the right arm here:

  * It reads ordinal pattern only -- which of the 24 points is bigger than which
    -- so it is blind to amplitude, offset and spread by construction.
  * A monotone window realises exactly ONE ordinal pattern. Its permutation
    entropy is then 0 for the same structural reason a constant vector has
    chaos 0: there is nothing to be entropic about. `pe_is_degenerate` catches
    that and the window is reported NOT RUN -- never a pass.

    THAT GUARD IS NOT ENOUGH ON ITS OWN, and an earlier version of this file
    claimed it was. `pe_is_degenerate` catches only EXACT monotonicity. Measured
    2026-08-26 (DooDoo, reproduced here): a straight line of slope 0.05/sample
    with N(0, 0.03) noise added -- 0.6x the per-step increment, which is what an
    ordinary drifting sensor looks like -- fired 12 of 12 windows at median
    z = -15 against a plain shuffle pool. One noisy sample creates a second
    ordinal pattern and the degeneracy guard stops applying, while the trend
    still dominates the statistic. The vulnerable band ran from sd = 0 up to
    roughly 2x the increment. Against a plain shuffle this arm was a drift
    detector, which is exactly what `harness/controls.py` documents at C3.

  * So the null is TREND-MATCHED, not a plain shuffle: each surrogate is the
    window's own OLS line plus a shuffled copy of its own residuals. The trend
    is preserved exactly and only the residual ordering is destroyed, so a
    trend cannot be the thing that fires. Neither an IAAFT pool nor a sorted
    surrogate does this job -- IAAFT still fired 10/10 at z = -12.03 on the same
    noisy ramp, and a sorted arrangement has permutation entropy 0, the minimum,
    so nothing can ever beat it.

MEASURED, NOT ASSUMED -- the controls, run through this exact code path
(see WORKFLOW_VERIFY_2026-08-26.md for the full tables):

    pure linear ramp        every window degenerate, 0 ran, 0 fired
    NOISY ramp, sd 0.02-0.15 x the per-step increment   0 of 5 seeds read
                            (against a plain shuffle this was 5 of 5 at z = -15)
    iid gaussian noise      ~5% of single windows fire, no channel reaches the
                            majority rule -> 0 channels read
    periodic (saw p8)       ~96% of windows fire -> 16 of 16 channels read
    sine, period 40         ~92% of windows fire -> 16 of 16 channels read

WHAT IT IS NOT
--------------
It is not the collapse engine. The collapse arm's own verdict on 24-point
windows is in the annex, and on this machine it has never returned READS for a
periodic input -- an exact sine is inside its own distribution-and-spectrum
null, which is a correct refusal, not a miss. Two different questions, reported
separately, neither standing in for the other.

lag-1 autocorrelation is carried alongside as a SEPARATE column and is never
part of the headline: a straight line passes lag-1 with the strongest possible
score. It measures trend and persistence, which is real but is not evidence of
anything beyond a trend.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from harness import perm_entropy, pe_is_degenerate  # noqa: E402

#: Points per window. 24 because that is what the rest of this stack reads.
PTS = 24
#: Shuffles per window. 500 gives a permutation-p resolution of 1/501 = 0.002.
NSHUF = 500
#: |z| a single window must reach to be counted a fire.
WIN_Z = 2.5
#: A channel reads only when a MAJORITY of the windows that ran fired, and at
#: least this many did. Measured on iid noise: single windows fire at 5.3%, so
#: a per-window threshold alone is not a channel-level answer. The best noise
#: channel reached 2 of 13.
MIN_FIRES = 3
#: Local windows per channel, evenly spaced across the record. A span window
#: (PTS points sampled across the WHOLE record) is added on top, because the
#: local windows only ever see PTS*LOCAL_WINDOWS consecutive rows.
#:
#: Adaptive between these bounds: a long file gets more windows so the coverage
#: line in the report is not embarrassing, a short one gets no more than its
#: rows can support. The report prints the number that was actually used and the
#: fraction of rows it touched -- never "every channel was screened".
LOCAL_WINDOWS = 12
MAX_WINDOWS = 40


def n_windows(n_rows):
    return int(min(MAX_WINDOWS, max(LOCAL_WINDOWS, n_rows // 100)))


def _z(observed, null):
    d = np.asarray(null, float)
    d = d[np.isfinite(d)]
    if d.size < 20:
        return None
    mu, sd = float(d.mean()), float(d.std())
    # A null with no spread cannot reject anything, and that is not a pass.
    if sd <= 1e-12 * max(1.0, abs(mu)):
        return None
    return float((observed - mu) / sd)


def _trend_pool(w, rng, n):
    """Surrogates that keep the window's trend and destroy everything else.

    Each draw is the window's own least-squares line plus a shuffled copy of its
    own residuals. The line is identical in every surrogate, so a trend is
    inside the null by construction and cannot be what fires the statistic. This
    is the drift arm; a plain shuffle pool is not one (see the module docstring).
    """
    w = np.asarray(w, float)
    i = np.arange(w.size, dtype=float)
    b, a = np.polyfit(i, w, 1)
    line = a + b * i
    res = w - line
    return [line + rng.permutation(res) for _ in range(n)]


def _lag1(w):
    x = np.asarray(w, float) - float(np.mean(w))
    den = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / den) if den > 1e-15 else 0.0


def window_starts(n, k=LOCAL_WINDOWS):
    """k evenly spaced starts for windows of PTS consecutive rows."""
    if n < PTS:
        return []
    return sorted(set(np.linspace(0, n - PTS, k).astype(int).tolist()))


def span_index(n):
    """PTS points sampled evenly ACROSS the whole record, not consecutively.

    Consecutive windows can only ever see PTS*LOCAL_WINDOWS rows. A span window
    sees the full record at coarse resolution, which is where slow structure
    lives. Both are reported; neither is presented as the other.
    """
    if n < PTS:
        return None
    idx = np.unique(np.linspace(0, n - 1, PTS).astype(int))
    return idx if idx.size == PTS else None


def screen_channel(x, rng, k=None):
    """One channel. Returns counts, never a bare verdict without them."""
    x = np.asarray(x, float)
    n = x.size
    if k is None:
        k = n_windows(n)
    wins = [("local", x[s:s + PTS]) for s in window_starts(n, k)]
    sp = span_index(n)
    if sp is not None:
        wins.append(("span", x[sp]))

    ran = fired = degenerate = unusable = trend = 0
    zs, rows_touched = [], set()
    for kind, w in wins:
        w = np.asarray(w, float)
        if not np.all(np.isfinite(w)) or float(np.std(w)) < 1e-12:
            unusable += 1
            continue
        perms = [rng.permutation(w) for _ in range(NSHUF)]
        # lag-1 is computed for EVERY usable window, including the monotone ones
        # the ordinal arm refuses, because a trend is exactly what it measures
        # and hiding it would be its own kind of dishonesty. It is scored against
        # the PLAIN shuffle pool on purpose -- it is the trend column, so the
        # trend must not be in its null.
        zl = _z(_lag1(w), [_lag1(p) for p in perms])
        if zl is not None and abs(zl) >= WIN_Z:
            trend += 1
        if pe_is_degenerate(w, 3):
            degenerate += 1
            continue
        # The headline arm is scored against the TREND-MATCHED pool, never the
        # plain shuffle. Against the shuffle it was a drift detector.
        pool = _trend_pool(w, rng, NSHUF)
        z = _z(perm_entropy(w, 3), [perm_entropy(p, 3) for p in pool])
        if z is None:
            unusable += 1
            continue
        ran += 1
        zs.append(z)
        if abs(z) >= WIN_Z:
            fired += 1

    for s in window_starts(n, k):
        rows_touched.update(range(s, min(s + PTS, n)))
    if sp is not None:
        rows_touched.update(int(i) for i in sp)

    reads = ran > 0 and fired >= max(MIN_FIRES, (ran + 1) // 2)
    return {
        "windows": len(wins), "ran": ran, "fired": fired,
        "degenerate": degenerate, "unusable": unusable, "trend": trend,
        "median_z": float(np.median(zs)) if zs else float("nan"),
        "rows_touched": len(rows_touched), "reads": bool(reads),
    }


def screen_all(X, names, rng, k=None):
    return [(nm, screen_channel(X[:, i], rng, k)) for i, nm in enumerate(names)]
