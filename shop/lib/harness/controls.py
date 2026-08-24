"""Four controls plus an engine-free baseline, applied to every read.

The reason this module is the point of the merge
------------------------------------------------
Every branch that died on this machine died the same way: a reading was taken
and the arm that could have killed it was missing, wrong, or structurally
incapable of failing. The ESM3 fusion "win" was a scorer padding masks to
alanine. The site-2 "algebra" control could not fail because the slots it
shuffled were near-fixed. The gas screen's permute test read a CONSTANT vector
as TEMPORAL because it failed open. None of those were engine faults.

So the controls live here, once, and `screen_window` refuses to return a
verdict unless they ran.

The four arms, and what each one kills
--------------------------------------
C0  flat degenerate   -- the INSTRUMENT gate. A constant vector must read
                         chaos_level 0. If it does not, the harness is void and
                         nothing else in the run may be quoted. Runs FIRST.
C1  own-shuffle       -- same 24 values, reordered. Kills "the engine sees
                         something" when the engine only sees the histogram.
C2  scale-matched     -- Gaussian vectors at the same norm. Kills a reading
                         that is merely typical of noise at that scale.
C3  sorted surrogate  -- THE SAME 24 VALUES in monotone order. MANDATORY: a
                         straight line with no periodicity at all scores
                         z = -3.97 TEMPORAL, so without a drift arm the screen
                         is a drift detector wearing a detector's name.

                         It is the sorted arrangement and not a fitted ramp
                         because a fitted ramp is not matched. Built as
                         slope*i + mean under the uncentred encoding, a ramp is
                         a nearly-constant vector with a tiny tilt, which sits
                         on the flat floor and scores -3.89 by being featureless
                         rather than by drifting. Measured 2026-08-23 on the
                         vetted GB-grid window: window -3.22, fitted ramp -3.89,
                         sorted surrogate -3.16. Sorting preserves the multiset,
                         the mean and the norm exactly and changes ONLY the
                         ordering, from real to monotone, which is the one thing
                         the arm is supposed to vary.
C5  IAAFT surrogate   -- the same values in the same power spectrum. C1
                         destroys the spectrum along with the ordering, so any
                         smooth series beats it; C3 patches the one worst shape
                         by hand. This is the matched null for smoothness in
                         general, and a sorted or shuffled arrangement of the
                         same numbers cannot pass it. Added 2026-08-23.
C6  permutation entropy - Bandt & Pompe ordinal patterns, orders 3 and 4,
                         scored against the IAAFT pool AND the shuffle pool.
                         Engine-free and blind to amplitude, DC offset and
                         spread, which are the three things the uncentred
                         collapse encoding actually responds to.
BASE engine-free      -- lag-1 autocorrelation and OLS slope through the SAME
                         permutation machinery. The collapse screen is blind to
                         AR(1) persistence (fires 4/20 on planted AR(1) where
                         lag-1 catches 19/20), so a null cannot be attributed
                         to the data rather than the instrument without it.

Two pools, and why one is not enough
------------------------------------
IAAFT preserves the autocorrelation function by construction, so linear
persistence -- AR(1) -- lies INSIDE the IAAFT null and nothing scored against
that pool can fire on it. Measured, not assumed (`arm_validation.py`, case
AR1). Both new arms are therefore scored against both pools: the IAAFT pool
answers "is there more here than the distribution and the spectrum", the
shuffle pool answers "is there any ordering information at all". Quote both or
neither.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from core import Reading, collapse24, prepare24  # noqa: E402
from core.collapse import DIM, verify_engine  # noqa: E402
from harness.surrogates import (N_SURROGATE, iaaft_pool, pe_is_degenerate,  # noqa: E402
                                perm_entropy, spectrum_fidelity)

#: |z| a reading must reach against its own shuffle to be considered at all.
PASS_Z = 2.5
#: |z| against the scale-matched arm.
RP_Z = 2.0
#: How far a reading must beat its own sorted surrogate before the difference
#: counts. Both z's are estimated from finite permutation samples, so a margin
#: of a hair is sampling noise -- and a hair is exactly what a monotone input
#: produced when this was a bare `>` comparison: own -4.63 vs sorted -4.63,
#: classified READS. A drift arm that a straight line can pass is not an arm.
DRIFT_MARGIN = 0.5
DEFAULT_SHUFFLES = 500
DEFAULT_RAMP = 300
DEFAULT_MATCHED = 300
#: IAAFT surrogates per reading (arms 5 and 6 share the pool). 200 gives a
#: two-sided permutation-p resolution of 1/201 = 0.005; no arm-5 or arm-6 p
#: below that is expressible, whatever the z says.
DEFAULT_IAAFT = N_SURROGATE
#: Ordinal-pattern orders for arm 6. Order 4 on a 24-point window yields 21
#: words over 24 possible patterns and is badly undersampled -- the entropy
#: estimate is biased low. The bias applies to the surrogates identically, so
#: the z is still valid; the POWER is what suffers. Read order 3 first.
PE_ORDERS = (3, 4)
#: All arms are read at one stated norm AND one stated centring, because both
#: change the answer and neither is recoverable from a quoted number.
#:
#: The defaults are the encoding the vetted GB-grid read was taken under
#: (`~/claude_collapse/permute_control.py`): norm 10, NOT mean-centred. That is
#: not a style choice. Measured 2026-08-23 on the Aug 11 window:
#:
#:      norm=10, uncentred   z = -3.37   TEMPORAL   (reproduces the vetted card)
#:      norm=10, centred     z = -0.45   invariant
#:      norm=1,  centred     z = -0.30   invariant
#:
#: Mean-centring DESTROYS that fire. The reading depends on the large constant
#: offset being present, so "period-8 fire on grid demand" is a property of the
#: encoding-with-DC-term, not of the oscillation alone. Any caller may change
#: these, but both travel with every result and no reading may be quoted
#: without them.
WORKING_NORM = 10.0
CENTRE = False


class Verdict(str, Enum):
    VOID = "VOID"                    # instrument gate failed; quote nothing
    FLAT = "FLAT"                    # input is constant; no direction to read
    INVARIANT = "INVARIANT"          # order does nothing. This is a result.
    DRIFT = "DRIFT"                  # reads, but its own sorted order reads as well or better
    SPECTRAL = "SPECTRAL"            # reads, but is inside its own distribution+spectrum null
    NOISE_LEVEL = "NOISE_LEVEL"      # not separated from scale-matched noise
    READS = "READS"                  # survived every arm


@dataclass(frozen=True)
class ArmResult:
    name: str
    z: float
    p_two_sided: float
    n: int
    resolution: float                 # smallest p this arm can express, 1/(n+1)
    skipped_reason: str | None = None

    @property
    def ran(self) -> bool:
        return self.skipped_reason is None


@dataclass(frozen=True)
class ScreenResult:
    verdict: Verdict
    reading: Reading | None
    arms: dict[str, ArmResult]
    baseline: dict[str, ArmResult]
    #: the encoding this reading was taken under -- never omit it when quoting
    norm: float = WORKING_NORM
    centre: bool = CENTRE
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def engine_free_hits(self) -> tuple[str, ...]:
        """Engine-free arms that reached PASS_Z: lag-1, OLS and arm 6."""
        return tuple(k for k, a in sorted(self.baseline.items())
                     if a.ran and abs(a.z) >= PASS_Z)

    @property
    def engine_adds_nothing(self) -> bool:
        """True when an engine-free arm reads and the collapse does not.

        Not a failure of the run -- it is the run's most useful output when it
        happens, because it says the structure is there and this instrument is
        the wrong one for it.

        The engine-free set is lag-1, OLS and arm 6 (permutation entropy,
        against both the IAAFT and the shuffle pool). Arm 6 joining this set is
        the reason it was added: it reads ordinal pattern only, so it can find
        sequence structure in exactly the place the uncentred collapse encoding
        -- which responds to amplitude, DC offset and spread -- is blind.
        """
        return bool(self.engine_free_hits) and self.verdict is not Verdict.READS

    @property
    def encoding(self) -> str:
        return f"norm={self.norm:g}, {'centred' if self.centre else 'uncentred'}"

    def line(self) -> str:
        z = {k: (f"{a.z:+.2f}" if a.ran else "skip") for k, a in self.arms.items()}
        b = {k: (f"{a.z:+.2f}" if a.ran else "skip") for k, a in self.baseline.items()}
        return (f"{self.verdict.value:<11} own {z.get('own_shuffle','-'):>6} "
                f"matched {z.get('scale_matched','-'):>6} "
                f"sorted {z.get('sorted_surrogate','-'):>6} "
                f"iaaft {z.get('iaaft','-'):>6} | "
                f"pe3 {b.get('pe3','-'):>6}/{b.get('pe3_shuf','-'):>6} "
                f"pe4 {b.get('pe4','-'):>6}/{b.get('pe4_shuf','-'):>6} "
                f"lag1 {b.get('lag1','-'):>6} ols {b.get('ols','-'):>6}"
                + ("  [ENGINE ADDS NOTHING: "
                   + ",".join(self.engine_free_hits) + "]"
                   if self.engine_adds_nothing else ""))


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def _z_and_p(observed: float, null: Sequence[float], name: str) -> ArmResult:
    """Two-sided z and permutation p against a null distribution.

    Two-sided on purpose. A one-sided p points at the wrong tail for half of
    these readings -- the collapse can separate in either direction and a
    fixed tail silently converts a real separation into a null.
    """
    d = np.asarray(null, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    res = 1.0 / (n + 1)
    if n < 20:
        return ArmResult(name, np.nan, np.nan, n, res,
                         skipped_reason=f"only {n} usable null draws")
    sd = float(d.std())
    mu = float(d.mean())
    # A null with no spread cannot reject anything. It is not a pass.
    #
    # The test is RELATIVE, and that is the whole point. It used to be the
    # absolute `sd < 1e-15`, which is not scale-free and let a genuinely
    # degenerate null through: 300 identical draws of a statistic near 6.63
    # accumulate float64 rounding to sd = 1.8e-15, just over the threshold, and
    # the observed value sits the same 1.8e-15 from the mean -- so the arm
    # returned a confident-looking z of EXACTLY 1.00 built entirely from
    # rounding error. At a statistic near 1000 the noise would be ~1e-12 and
    # would sail past unnoticed. Found 2026-08-23 on a fully confounded
    # population design, where every surrogate is the same partition relabelled.
    scale = max(1.0, abs(mu))
    if sd <= 1e-12 * scale or float(np.ptp(d)) <= 1e-12 * scale:
        return ArmResult(name, np.nan, np.nan, n, res,
                         skipped_reason=(
                             f"null has no usable spread (sd={sd:.3e}, "
                             f"range={float(np.ptp(d)):.3e}, relative to a "
                             f"scale of {scale:.3g}) -- every draw is the same "
                             "value to within floating-point rounding, so this "
                             "arm cannot fail and is reported as not run"))
    z = (observed - mu) / sd
    p = (1 + int(np.sum(np.abs(d - mu) >= abs(observed - mu)))) / (n + 1)
    return ArmResult(name, float(z), float(p), n, res)


def _chaos(window: np.ndarray, *, centre: bool, norm: float) -> float:
    v = prepare24(window, centre=centre, norm=norm)
    if not np.any(v):
        return 0.0
    return collapse24(v).chaos_level


def _lag1(window: np.ndarray) -> float:
    x = np.asarray(window, float) - np.mean(window)
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 1e-15 else 0.0


def _ols(window: np.ndarray) -> float:
    return float(np.polyfit(np.arange(len(window)), window, 1)[0])


# --------------------------------------------------------------------------
# C0 -- the instrument gate
# --------------------------------------------------------------------------

def instrument_gate() -> tuple[bool, str]:
    """Run the flat degenerate control. Must pass before anything else.

    A constant vector has no ordering information whatsoever, so its
    chaos_level must be 0. A harness that returns anything else is failing
    open, and every downstream verdict from it is uninterpretable -- this
    exact defect once read a constant vector as TEMPORAL.
    """
    verify_engine()
    flat = np.full(DIM, 1.0 / np.sqrt(DIM))
    c = collapse24(flat).chaos_level
    ok = abs(c) < 1e-9
    return ok, (f"flat degenerate chaos_level={c:.3e} -> "
                f"{'PASS' if ok else 'VOID -- harness fails open, quote nothing'}")


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------

def screen_window(window: Sequence[float],
                  *,
                  rng: np.random.Generator,
                  n_shuffle: int = DEFAULT_SHUFFLES,
                  n_ramp: int = DEFAULT_RAMP,
                  n_matched: int = DEFAULT_MATCHED,
                  n_iaaft: int = DEFAULT_IAAFT,
                  centre: bool = CENTRE,
                  norm: float = WORKING_NORM,
                  gate_checked: bool = False) -> ScreenResult:
    """Screen one 24-point ordered window through every arm.

    There is no opt-out. Arms 5 and 6 run on the same call as C0-C3, so a
    reading cannot exist without them -- which is the whole reason they live
    here and not in a script beside the harness.

    `gate_checked=True` says the caller already ran `instrument_gate` for this
    process. Sweeps should do that once rather than 40,000 times; a single
    channel screen should leave it False.
    """
    notes: list[str] = []
    if not gate_checked:
        ok, msg = instrument_gate()
        notes.append(msg)
        if not ok:
            return ScreenResult(Verdict.VOID, None, {}, {}, norm, centre, tuple(notes))

    w = np.asarray(window, dtype=float)
    if w.shape != (DIM,):
        raise ValueError(f"screen_window takes exactly {DIM} points, got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("non-finite value in the window")
    if float(np.std(w)) < 1e-12:
        notes.append("input is constant -- flat degenerate, nothing to order")
        return ScreenResult(Verdict.FLAT, None, {}, {}, norm, centre, tuple(notes))

    reading = collapse24(prepare24(w, centre=centre, norm=norm))
    observed = reading.chaos_level
    ch = lambda x: _chaos(x, centre=centre, norm=norm)          # noqa: E731

    arms: dict[str, ArmResult] = {}

    # C1 -- own shuffle: identical multiset, different order.
    arms["own_shuffle"] = _z_and_p(
        observed, [ch(rng.permutation(w)) for _ in range(n_shuffle)],
        "own_shuffle")

    # C2 -- Gaussian surrogates matched in MEAN and SPREAD, not just norm.
    # Matching the norm alone is not a control when the reading is taken
    # uncentred: a zero-mean Gaussian rescaled to the same norm has none of the
    # DC component the real vector has, and the DC component is exactly what
    # the uncentred encoding reads. That would compare two different encodings
    # and call the difference a signal.
    mu, sd = float(np.mean(w)), float(np.std(w))
    arms["scale_matched"] = _z_and_p(
        observed,
        [ch(mu + sd * rng.standard_normal(DIM)) for _ in range(n_matched)],
        "scale_matched")

    # C3 -- the same values in monotone order: exactly matched drift surrogate.
    # Ascending and descending are both monotone; take whichever reads stronger,
    # so the arm cannot be passed by an accident of sign.
    asc = np.sort(w)
    cands = []
    for surrogate in (asc, asc[::-1]):
        cands.append(_z_and_p(
            ch(surrogate),
            [ch(rng.permutation(surrogate)) for _ in range(n_ramp // 2)],
            "sorted_surrogate"))
    ran = [c for c in cands if c.ran]
    arms["sorted_surrogate"] = (max(ran, key=lambda c: abs(c.z)) if ran
                                else cands[0])

    # C5 -- IAAFT: same values EXACTLY, power spectrum matched to convergence.
    # This is the arm C1 and C3 could not be. C1 destroys the spectrum, so any
    # smooth series beats it. C3 fixes one shape, monotone, by hand. Here the
    # null already contains the distribution and the spectrum, so a sorted or
    # shuffled arrangement of the same numbers has nothing left to win with.
    pool = iaaft_pool(w, rng, n=n_iaaft)
    fid = spectrum_fidelity(w, pool)
    arms["iaaft"] = _z_and_p(observed, [ch(s) for s in pool], "iaaft")
    # A DEGENERATE IAAFT pool is not missing evidence -- it is the strongest
    # possible form of the SPECTRAL finding. If every surrogate that matches
    # this window's distribution and power spectrum also reproduces its
    # reading, then the window IS its distribution and spectrum and there is
    # nothing else in it. An exact sine is the standing example: it has one
    # Fourier component, so its whole surrogate pool is the same object again.
    iaaft_degenerate = (not arms["iaaft"].ran
                        and "no usable spread" in (arms["iaaft"].skipped_reason or ""))
    if iaaft_degenerate:
        notes.append("IAAFT pool is DEGENERATE: every distribution- and "
                     "spectrum-matched surrogate reproduces this reading, so "
                     "the window IS its distribution and spectrum. That is a "
                     "stronger SPECTRAL result than a low z, not a missing arm.")
    notes.append(f"IAAFT pool n={len(pool)}, mean spectral error "
                 f"{fid:.4f} of ||amp||; exact value distribution by "
                 f"construction")
    if np.isfinite(fid) and fid > 0.25:
        notes.append(f"IAAFT spectral error {fid:.3f} is large -- the pool is "
                     "not well matched to this window and arm 5 should not be "
                     "quoted for it")

    # BASE -- engine-free, same permutation machinery.
    perms = [rng.permutation(w) for _ in range(n_shuffle)]
    baseline = {
        "lag1": _z_and_p(_lag1(w), [_lag1(p) for p in perms], "lag1"),
        "ols": _z_and_p(_ols(w), [_ols(p) for p in perms], "ols"),
    }

    # C6 -- permutation entropy, ordinal pattern only. Scored against BOTH
    # pools on purpose: IAAFT keeps the autocorrelation function, so linear
    # persistence is inside that null and only the shuffle pool can see it.
    for order in PE_ORDERS:
        keys = (f"pe{order}", f"pe{order}_shuf")
        if pe_is_degenerate(w, order):
            why = (f"window realises ONE ordinal pattern at order {order} "
                   "(monotone) -- permutation entropy is 0 by construction, "
                   "which is degenerate, not a hit")
            for k in keys:
                baseline[k] = ArmResult(k, np.nan, np.nan, 0, np.nan,
                                        skipped_reason=why)
            continue
        pe = perm_entropy(w, order)
        baseline[keys[0]] = _z_and_p(pe, [perm_entropy(s, order) for s in pool],
                                     keys[0])
        baseline[keys[1]] = _z_and_p(pe, [perm_entropy(p, order) for p in perms],
                                     keys[1])

    own = arms["own_shuffle"]
    matched = arms["scale_matched"]
    rampa = arms["sorted_surrogate"]
    iaaft = arms["iaaft"]

    # A monotone window IS its own sorted surrogate. No statistic is needed to
    # say so, and none should be trusted to: the two z's are then estimates of
    # the same quantity and their difference is noise.
    order = np.argsort(np.argsort(w))
    monotone = bool(np.all(np.diff(w) >= 0) or np.all(np.diff(w) <= 0))
    del order
    if not own.ran:
        notes.append(f"own-shuffle arm did not run: {own.skipped_reason}")
        verdict = Verdict.VOID
    elif abs(own.z) < PASS_Z:
        verdict = Verdict.INVARIANT
    elif monotone:
        notes.append("window is exactly monotone -- it IS its own sorted "
                     "surrogate, so this is drift by construction")
        verdict = Verdict.DRIFT
    elif rampa.ran and abs(own.z) <= abs(rampa.z) + DRIFT_MARGIN:
        verdict = Verdict.DRIFT
    elif iaaft_degenerate or (iaaft.ran and abs(iaaft.z) < PASS_Z):
        # Beat its own shuffle but sits inside a null that already holds its
        # value distribution and its power spectrum. Whatever it is responding
        # to, a linear Gaussian process with the same second-order structure
        # reproduces it. That is not a detection.
        notes.append("beats its own shuffle but not its IAAFT pool: the "
                     "reading is explained by the value distribution plus the "
                     "power spectrum")
        verdict = Verdict.SPECTRAL
    elif matched.ran and abs(matched.z) < RP_Z:
        verdict = Verdict.NOISE_LEVEL
    else:
        verdict = Verdict.READS
        if not rampa.ran:
            notes.append("READS without a drift arm -- weaker claim, say so")
        if not matched.ran:
            notes.append("READS without a scale-matched arm -- weaker claim")
        if not iaaft.ran and not iaaft_degenerate:
            notes.append("READS without the IAAFT arm -- weaker claim; the "
                         "distribution+spectrum confound is untested here")

    # The chain returns the FIRST reason a window was rejected, but a window
    # can fail several arms at once and the single label then hides the rest.
    # An exact period-8 sine is the standing example: it does not beat its own
    # sorted arrangement (DRIFT) AND it does not beat its own IAAFT pool
    # (SPECTRAL). Both are true, only one becomes the label, so the other is
    # recorded here rather than lost.
    if verdict in (Verdict.DRIFT, Verdict.SPECTRAL, Verdict.NOISE_LEVEL):
        also = []
        if verdict is not Verdict.DRIFT and rampa.ran and \
                abs(own.z) <= abs(rampa.z) + DRIFT_MARGIN:
            also.append("does not beat its own sorted surrogate (DRIFT)")
        if verdict is not Verdict.SPECTRAL and (
                iaaft_degenerate or (iaaft.ran and abs(iaaft.z) < PASS_Z)):
            also.append("is inside its own distribution+spectrum null (SPECTRAL)")
        if verdict is not Verdict.NOISE_LEVEL and matched.ran and \
                abs(matched.z) < RP_Z:
            also.append("is not separated from scale-matched noise (NOISE_LEVEL)")
        if also:
            notes.append(f"also rejected because it {'; and it '.join(also)}")

    if own.ran and abs(own.z) >= PASS_Z and reading.control_norm < 0:
        notes.append("control_norm is negative: the input is approaching "
                     "featureless, which is what TEMPORAL means here -- it is "
                     "not evidence of structure")

    return ScreenResult(verdict, reading, arms, baseline, norm, centre,
                        tuple(notes))
