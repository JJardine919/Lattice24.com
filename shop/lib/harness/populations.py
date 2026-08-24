"""The POPULATION path: screen a labelled comparison between groups.

`screen_window` asks "does the ORDER of these 24 numbers carry information".
That is one of two questions this box has ever asked, and it is not the one the
only surviving result answers. Gas separates six populations by entropy
signature; PG(5,2) separates two classes of magic square. Neither is a sequence
question, and until now neither had a harness -- so each was screened by a
hand-rolled script, and the hand-rolled scripts kept making the same mistake.

THE MISTAKE, and why the assertion below is not decoration
----------------------------------------------------------
In `pg52_screen.py` (2026-08-23) the engine arm was fed
    row / ||row|| * 10          -- per-row L2 normalised, row magnitude DESTROYED
and the matched random-projection arm was fed
    (X - X.mean(0)) / X.std(0)  -- column z-scores, row magnitude RETAINED
Two arms, two different datasets, one comparison drawn between them. The RP arm
then "beat the engine" (120 of 200 matched or beat AUC 0.651) partly on row-norm
information the engine had been denied -- row norm alone scores 0.683. That is
not a fair loss and it is not a fair win either; it is not a measurement.

So every arm in this module is handed THE SAME preprocessed matrix, and the
matrix is fingerprinted before the first arm runs and re-checked inside each
one. An arm that receives different bytes raises rather than reports.

    A consequence worth stating out loud: the engine REQUIRES normalisation, so
    the matched comparison is necessarily run on data with row magnitude removed
    -- from BOTH arms. When row norm is itself predictive, that is information
    the engine's own input contract forbids it from using. That is a real
    limitation of the engine, not an artefact of the harness, and it is reported
    separately as `row_norm_denied` rather than smuggled into either arm.

THE ARMS
--------
P1  label shuffle    -- shuffle which unit belongs to which population, keep
                        everything else. If shuffled labels reach the observed
                        separation, the statistic is counting n, not separation.

                        P1 ASSUMES THE UNITS ARE EXCHANGEABLE. When they are
                        not -- overlapping rolling windows are the standard
                        case -- a unit-level shuffle scatters each group's
                        correlated neighbours across every surrogate group and
                        DESTROYS the dependence the real design has. The null
                        collapses and the z inflates. Measured on gas
                        (2026-08-23): a block-preserving shuffle gives a null of
                        7.1 +- 2.1 and p = 0.0033 for the observed 18.44, while
                        the unit-level shuffle gives 1.86 +- 0.64 and z = +31.28
                        for the same data. The +31.28 is ANTI-CONSERVATIVE and
                        is not stronger evidence than the 0.0033.
                        Pass `blocks=` to shuffle at the block level instead.
                        If you do not, the harness measures the unit-to-unit
                        dependence itself and says so.
P2  matched RP       -- random projections of the SAME matrix to the SAME number
                        of channels the engine produced, scored the SAME way.
                        Chance is the weak comparator; this is the real bar.
BASE best_column     -- the single best raw column, scored identically. The
                        population-path equivalent of lag-1.
BASE t_stat          -- plain Welch t between populations. A second, parametric
                        engine-free comparator, so a rank statistic and a moment
                        statistic both have to be beaten.
Every arm is scored against the P1 null, so [ENGINE ADDS NOTHING] fires on
exactly the rule the order path uses: an engine-free arm reads and the engine
verdict is not SEPARATES.
"""
from __future__ import annotations

import hashlib
import itertools
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from core import collapse24, prepare24                      # noqa: E402
from core.collapse import DIM                               # noqa: E402
from harness.controls import (ArmResult, PASS_Z, WORKING_NORM,  # noqa: E402
                              _z_and_p, instrument_gate)

DEFAULT_LABEL_SHUFFLES = 300
DEFAULT_RP = 200
#: Fraction of matched random projections allowed to match or beat the engine
#: before the result is called RP-dominated. 5% of the pool, i.e. the engine
#: must sit in the top 5% of arbitrary summaries of the same data.
RP_DOMINATED_FRAC = 0.05
#: Collapse channels used as the engine's feature vector. The RP arm is given
#: exactly this many channels, so the max-over-channels inflation is matched.
CHANNELS = ("chaos_level", "normalized_chaos", "intent_magnitude", "control_norm")


class PopVerdict(str, Enum):
    VOID = "VOID"                      # instrument gate or a null failed; quote nothing
    FLAT = "FLAT"                      # no variance to compare; nothing to separate
    NO_SEPARATION = "NO_SEPARATION"    # does not beat its own shuffled labels
    RP_DOMINATED = "RP_DOMINATED"      # beats chance, loses to arbitrary projections
    TRIVIAL = "TRIVIAL"                # an engine-free column does it as well or better
    SEPARATES = "SEPARATES"            # survived every arm


# --------------------------------------------------------------------------
# preprocessing -- ONE function, applied ONCE, shared by every arm
# --------------------------------------------------------------------------

def _span(a: np.ndarray, k: int = DIM) -> np.ndarray:
    return a[:, np.linspace(0, a.shape[1] - 1, k).round().astype(int)]


def preprocess(X: Sequence[Sequence[float]], *, mode: str = "engine",
               norm: float = WORKING_NORM, centre: bool = False) -> np.ndarray:
    """Produce the ONE matrix every arm will see.

    mode="engine" is the collapse input contract: span to 24 columns if wider,
    then normalise each row to `norm`. Rows that cannot be normalised (zero
    variance under `centre`, or all-zero) come back as zeros and are reported,
    never silently dropped.
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"screen_populations takes a 2-D matrix, got {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("non-finite value in the population matrix")
    if mode == "none":
        return A.copy()
    if mode == "zscore":
        sd = A.std(0)
        return (A - A.mean(0)) / np.where(sd > 0, sd, 1.0)
    if mode != "engine":
        raise ValueError(f"unknown preprocess mode {mode!r}")
    if A.shape[1] > DIM:
        A = _span(A)
    if A.shape[1] != DIM:
        raise ValueError(
            f"the engine takes exactly {DIM} columns, got {A.shape[1]}. "
            "Downsample to 24 -- never switch engines to dodge the constraint.")
    return np.vstack([prepare24(r, centre=centre, norm=norm) for r in A])


def _unit_dependence(F: np.ndarray) -> float:
    """Lag-1 autocorrelation of the feature vector in row order.

    A cheap, engine-free exchangeability check. High values mean consecutive
    units share information -- overlapping windows, repeated measures, adjacent
    time steps -- and a unit-level permutation null is then anti-conservative.
    It cannot prove exchangeability; it can only refuse to stay silent when the
    data plainly is not.
    """
    if F.shape[0] < 8:
        return float("nan")
    v = np.asarray(F[:, 0], float)
    v = v - v.mean()
    d = float(v @ v)
    return abs(float(v[:-1] @ v[1:] / d)) if d > 1e-15 else float("nan")


def _block_shuffler(y: np.ndarray, blocks, rng: np.random.Generator):
    """Return a callable producing one surrogate label vector per call."""
    if blocks is None:
        return lambda: rng.permutation(y)
    b = np.asarray(blocks)
    uniq = np.unique(b)
    idx = [np.flatnonzero(b == u) for u in uniq]
    # One label per block, taken from the block's own majority, then the block
    # labels are permuted. Block sizes and within-block dependence are both
    # preserved exactly; only which block belongs to which population moves.
    lab = np.array([np.bincount(
        np.searchsorted(np.unique(y), y[i])).argmax() for i in idx])
    groups = np.unique(y)

    def draw():
        out = np.empty_like(y)
        for i, L in zip(idx, rng.permutation(lab)):
            out[i] = groups[L]
        return out
    return draw


def _fingerprint(A: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(A, dtype=np.float64).tobytes()).hexdigest()


class _Guarded:
    """Hands out the shared matrix and refuses if anyone changed the bytes.

    This is the assertion the pg52 defect asks for. It is not a comment, it is
    not a convention, and it cannot be satisfied by two arms that merely look
    like they preprocess the same way.
    """

    def __init__(self, Xp: np.ndarray):
        self._X = np.ascontiguousarray(Xp, dtype=np.float64)
        self._X.flags.writeable = False
        self.fp = _fingerprint(self._X)

    def get(self, who: str) -> np.ndarray:
        got = _fingerprint(self._X)
        if got != self.fp:
            raise RuntimeError(
                f"arm {who!r} was about to receive a DIFFERENT matrix "
                f"({got} != {self.fp}). Every arm must see identical bytes -- "
                "this exact mismatch is what made the pg52 RP comparison "
                "meaningless.")
        return self._X


# --------------------------------------------------------------------------
# separation statistics -- one function, applied to engine, RP and raw alike
# --------------------------------------------------------------------------

def _auc(score: np.ndarray, pos: np.ndarray) -> float:
    o = np.argsort(score, kind="mergesort")
    r = np.empty(score.size, float)
    r[o] = np.arange(1, score.size + 1, dtype=float)
    s = score[o]
    i = 0
    while i < s.size:                       # average ranks over ties
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = r[o[i:j + 1]].mean()
        i = j + 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_auc(F: np.ndarray, y: np.ndarray) -> float:
    """max |AUC - 0.5| + 0.5 over columns. Two groups only. 0.5 = nothing."""
    groups = np.unique(y)
    pos = y == groups[-1]
    best = 0.5
    for k in range(F.shape[1]):
        c = F[:, k]
        if float(np.std(c)) < 1e-15:
            continue
        best = max(best, abs(_auc(c, pos) - 0.5) + 0.5)
    return best


def score_maxz(F: np.ndarray, y: np.ndarray) -> float:
    """max |two-sample Welch z| over columns AND population pairs.

    For two groups and one column this IS the t-statistic, so the same code
    path serves the six-population gas comparison and a two-class AUC-free
    comparison without a second implementation to keep honest.
    """
    groups = list(np.unique(y))
    best = 0.0
    for k in range(F.shape[1]):
        c = F[:, k]
        for a, b in itertools.combinations(groups, 2):
            A, B = c[y == a], c[y == b]
            if A.size < 5 or B.size < 5:
                continue
            se = np.sqrt(A.var(ddof=1) / A.size + B.var(ddof=1) / B.size)
            if se > 1e-15:
                best = max(best, abs(float(A.mean() - B.mean())) / se)
    return best


SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auc": score_auc, "maxz": score_maxz}


# --------------------------------------------------------------------------
# the engine's own featurisation
# --------------------------------------------------------------------------

def collapse_features(Xp: np.ndarray) -> np.ndarray:
    """Collapse every row of the SHARED matrix into the channel vector."""
    out = np.empty((Xp.shape[0], len(CHANNELS)))
    for i, row in enumerate(Xp):
        if not np.any(row):
            out[i] = 0.0                    # flat degenerate row, reported upstream
            continue
        r = collapse24(row)
        out[i] = [getattr(r, c) for c in CHANNELS]
    return out


# --------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PopResult:
    verdict: PopVerdict
    n: int
    n_groups: int
    group_sizes: dict
    separation: float
    stat: str
    arms: dict[str, ArmResult]
    baseline: dict[str, ArmResult]
    observed: dict[str, float]
    rp_beat: int = 0
    rp_n: int = 0
    preprocess_mode: str = "engine"
    fingerprint: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def engine_free_hits(self) -> tuple[str, ...]:
        return tuple(k for k, a in sorted(self.baseline.items())
                     if a.ran and abs(a.z) >= PASS_Z)

    @property
    def engine_adds_nothing(self) -> bool:
        """Identical rule to the order path: a free arm reads, the engine does not."""
        return (bool(self.engine_free_hits)
                and self.verdict is not PopVerdict.SEPARATES)

    def line(self) -> str:
        a = {k: (f"{v.z:+.2f}" if v.ran else "skip")
             for k, v in self.arms.items()}
        b = {k: (f"{v.z:+.2f}" if v.ran else "skip")
             for k, v in self.baseline.items()}
        return (f"{self.verdict.value:<14} sep {self.separation:>7.3f} "
                f"({self.stat}) | shuffle {a.get('label_shuffle','-'):>7} "
                f"| RP {self.rp_beat}/{self.rp_n} beat, z {a.get('matched_rp','-'):>7} "
                f"| col {self.observed.get('best_column', float('nan')):>7.3f} "
                f"z {b.get('best_column','-'):>7} "
                f"| t {b.get('t_stat','-'):>7}"
                + ("  [ENGINE ADDS NOTHING: " + ",".join(self.engine_free_hits) + "]"
                   if self.engine_adds_nothing else ""))


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------

def screen_populations(X, labels, *,
                       rng: np.random.Generator,
                       featurise: Callable[[np.ndarray], np.ndarray] = collapse_features,
                       stat: str | None = None,
                       mode: str = "engine",
                       norm: float = WORKING_NORM,
                       centre: bool = False,
                       n_shuffle: int = DEFAULT_LABEL_SHUFFLES,
                       n_rp: int = DEFAULT_RP,
                       blocks=None,
                       trivial_match_k: bool = False,
                       gate_checked: bool = False,
                       raw_for_rownorm=None) -> PopResult:
    """Screen a labelled comparison between populations.

    Returns a verdict on the same footing as `screen_window`: it refuses to
    produce one unless every arm ran, and a null with no spread is reported as
    NOT RUN rather than passed.
    """
    notes: list[str] = []
    if not gate_checked:
        ok, msg = instrument_gate()
        notes.append(msg)
        if not ok:
            return PopResult(PopVerdict.VOID, 0, 0, {}, float("nan"), "-",
                             {}, {}, {}, notes=tuple(notes))

    y = np.asarray(labels)
    groups, counts = np.unique(y, return_counts=True)
    sizes = {str(g): int(c) for g, c in zip(groups, counts)}
    if groups.size < 2:
        notes.append(f"only {groups.size} population(s) -- nothing to compare")
        return PopResult(PopVerdict.FLAT, len(y), int(groups.size), sizes,
                         float("nan"), "-", {}, {}, {}, notes=tuple(notes))

    stat = stat or ("auc" if groups.size == 2 else "maxz")
    if stat not in SCORERS:
        raise ValueError(f"unknown stat {stat!r}; use one of {sorted(SCORERS)}")
    score = SCORERS[stat]

    Xp = preprocess(X, mode=mode, norm=norm, centre=centre)
    if Xp.shape[0] != y.size:
        raise ValueError(f"{Xp.shape[0]} rows but {y.size} labels")
    dead = int(np.sum(~np.any(Xp, axis=1)))
    if dead:
        notes.append(f"{dead} of {Xp.shape[0]} rows are flat degenerate after "
                     "preprocessing and contribute a zero feature vector")
    if float(np.std(Xp)) < 1e-15:
        notes.append("preprocessed matrix has no variance -- nothing to separate")
        return PopResult(PopVerdict.FLAT, len(y), int(groups.size), sizes,
                         float("nan"), stat, {}, {}, {},
                         preprocess_mode=mode, notes=tuple(notes))

    guard = _Guarded(Xp)
    notes.append(f"all arms share matrix {Xp.shape} fingerprint {guard.fp[:12]} "
                 f"(preprocess={mode}, norm={norm:g}, "
                 f"{'centred' if centre else 'uncentred'})")

    # ---- the engine ------------------------------------------------------
    F = featurise(guard.get("engine"))
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    if F.shape[0] != y.size:
        raise ValueError(f"featurise returned {F.shape[0]} rows for {y.size} labels")
    observed = float(score(F, y))
    k = F.shape[1]

    arms: dict[str, ArmResult] = {}
    baseline: dict[str, ArmResult] = {}

    # ---- P1: shuffle which unit belongs to which population --------------
    # At the BLOCK level when the caller says the units are not exchangeable,
    # so each surrogate group keeps the dependence structure the real one has.
    shuffle = _block_shuffler(y, blocks, rng)
    null_shuffle = [score(F, shuffle()) for _ in range(n_shuffle)]
    arms["label_shuffle"] = _z_and_p(observed, null_shuffle, "label_shuffle")
    if blocks is not None:
        notes.append(f"P1 shuffled at the BLOCK level "
                     f"({len(np.unique(np.asarray(blocks)))} blocks), so each "
                     "surrogate keeps the within-block dependence")
    else:
        dep = _unit_dependence(F)
        if np.isfinite(dep) and dep >= 0.5:
            notes.append(
                f"UNITS LOOK NON-EXCHANGEABLE: lag-1 dependence between "
                f"consecutive rows is {dep:.2f}. A unit-level P1 shuffle "
                "scatters correlated neighbours across every surrogate group, "
                "which DESTROYS that dependence, collapses the null and "
                "INFLATES this z. Overlapping rolling windows are the standard "
                "cause. Pass blocks= to shuffle at the block level; until then "
                "treat the P1 z as anti-conservative and quote a "
                "block-preserving p instead.")

    # ---- P2: matched random projections, SAME matrix, SAME k, SAME scorer -
    Xr = guard.get("matched_rp")
    rp_scores = []
    for _ in range(n_rp):
        W = rng.standard_normal((Xr.shape[1], k))
        rp_scores.append(score(Xr @ W, y))
    rp_scores = np.asarray(rp_scores, float)
    rp_beat = int(np.sum(rp_scores >= observed))
    arms["matched_rp"] = _z_and_p(observed, rp_scores, "matched_rp")

    # ---- BASE: trivial engine-free comparators, SAME matrix, SAME scorer --
    Xb = guard.get("baseline")
    col_scores = np.array([score(Xb[:, [j]], y) for j in range(Xb.shape[1])])
    best_j = int(np.argmax(col_scores))
    best_col = float(col_scores[best_j])

    # MATCHED AT k. `best_col` is the max over ALL columns while the engine's
    # score is the max over its k channels, so the comparator carries more
    # selection inflation than the thing it is compared against -- the RP arm
    # is matched at k and this one was not. With trivial_match_k the TRIVIAL
    # rule uses the median, over random k-column subsets, of the best column
    # WITHIN the subset: the same max-over-k inflation the engine gets.
    # OFF BY DEFAULT ON PURPOSE. Turning it on changes verdicts, and every
    # domain JSON already on disk was produced with it off; silently
    # re-scoring those would be worse than the defect. The acceptance battery
    # turns it on. Reported either way as `best_of_k`.
    best_of_k = best_col
    if trivial_match_k and 0 < k < Xb.shape[1]:
        subs = [float(np.max(col_scores[rng.choice(Xb.shape[1], k, replace=False)]))
                for _ in range(200)]
        best_of_k = float(np.median(subs))
    # Scored against the SAME P1 null, so its z is on the engine's footing.
    base_null = [score(Xb[:, [best_j]], rng.permutation(y)) for _ in range(n_shuffle)]
    baseline["best_column"] = _z_and_p(best_col, base_null, "best_column")

    t_obs = score_maxz(Xb, y)
    t_null = [score_maxz(Xb, rng.permutation(y)) for _ in range(min(n_shuffle, 150))]
    baseline["t_stat"] = _z_and_p(t_obs, t_null, "t_stat")

    obs = dict(engine=observed, best_column=best_col, best_column_index=best_j,
               best_of_k=best_of_k, trivial_match_k=bool(trivial_match_k),
               t_stat=t_obs, rp_mean=float(rp_scores.mean()),
               rp_sd=float(rp_scores.std()), channels=k)

    # ---- the information the engine's own contract denies it -------------
    # Row magnitude is destroyed by the normalisation the engine REQUIRES, and
    # it is destroyed for every arm equally, so the comparison above is fair.
    # But if row norm predicts the label, that is a real thing the engine
    # cannot use, and hiding it inside a "fair" comparison would be a lie of
    # omission. It is reported here and never folded into a verdict.
    src = np.asarray(raw_for_rownorm if raw_for_rownorm is not None else X, float)
    if src.ndim == 2 and src.shape[0] == y.size and mode == "engine":
        rn = np.linalg.norm(src, axis=1)[:, None]
        if float(np.std(rn)) > 1e-15:
            rn_score = float(score(rn, y))
            obs["row_norm_denied"] = rn_score
            if rn_score >= observed:
                notes.append(
                    f"ROW NORM ALONE scores {rn_score:.3f} against the engine's "
                    f"{observed:.3f}. The engine cannot use it: normalisation is "
                    "its input contract. This is a limitation of the engine, not "
                    "a defect of this comparison -- both arms above were denied "
                    "it equally.")

    # ---- verdict ---------------------------------------------------------
    ls, rp = arms["label_shuffle"], arms["matched_rp"]
    if not ls.ran:
        notes.append(f"label-shuffle arm did not run: {ls.skipped_reason}")
        verdict = PopVerdict.VOID
    elif abs(ls.z) < PASS_Z:
        verdict = PopVerdict.NO_SEPARATION
    elif rp_beat > RP_DOMINATED_FRAC * n_rp:
        notes.append(f"{rp_beat} of {n_rp} arbitrary {Xr.shape[1]}->{k} random "
                     "projections of the SAME matrix match or beat the engine")
        verdict = PopVerdict.RP_DOMINATED
    elif best_of_k >= observed or (stat == "maxz" and t_obs >= observed):
        # SCALE BUG, fixed 2026-08-23. This read
        #     best_col >= observed or t_obs >= observed
        # unconditionally. `best_col` is produced by the SAME scorer as the
        # engine, so that half is a like-for-like comparison. `t_obs` is
        # `score_maxz` ALWAYS, whatever `stat` is -- so under stat="auc" it
        # compared a Welch t-statistic (tens) against an AUC (at most 1.0) and
        # was true whenever the t exceeded 1.0, i.e. essentially always.
        #
        # Consequence while it stood: with stat="auc", SEPARATES was
        # unreachable BY CONSTRUCTION. Measured on the ceiling run -- a
        # synthetic comparison at 30 sd of shape shift where the engine scored
        # AUC 0.999, only 2 of 200 random projections matched it, and the best
        # raw column reached 0.867 -- was labelled TRIVIAL. That is a verdict
        # about units, not about data, and it would have been reported as
        # "the engine has no reachable positive verdict".
        #
        # t_obs stays in the comparison where it IS commensurable (stat="maxz",
        # which is what the pooled six-gas run uses, so `pop_gas.json` is
        # unaffected) and is reported as a z everywhere else.
        why = (f"the best single raw column ({best_of_k:.3f}"
               + (f", matched at k={k}" if trivial_match_k
                  else f", col {best_j}") + ")"
               if best_of_k >= observed else f"a plain Welch t ({t_obs:.3f})")
        notes.append(f"engine {observed:.3f} does not beat {why}")
        verdict = PopVerdict.TRIVIAL
    else:
        verdict = PopVerdict.SEPARATES
        if not rp.ran:
            notes.append("SEPARATES without a usable RP z -- weaker claim, say so")

    return PopResult(verdict, int(y.size), int(groups.size), sizes, observed,
                     stat, arms, baseline, obs, rp_beat, n_rp, mode,
                     guard.fp, tuple(notes))


# --------------------------------------------------------------------------
# the dose ladder
# --------------------------------------------------------------------------

def dose_ladder(X, labels, *, rng: np.random.Generator,
                doses: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                n_rep: int = 5, **kw) -> dict:
    """Degrade the LABELS by a known amount and watch the separation fall.

    A passing positive control proves nothing on its own -- it says the screen
    can return a hit, not that the hit tracks the thing it claims to measure.
    Monotone dose-response is what distinguishes those, so it is required here
    rather than offered.

    `dose` is the fraction of units whose labels are PERMUTED AMONG THEMSELVES.
    At dose 0 the labels are the real ones; at dose 1 the ladder degenerates to
    a full permutation, which is exactly the P1 null -- so the last rung is not
    merely "noise", it is the same null the label-shuffle arm uses, and the
    ladder is required to land on it.

    Permuting rather than resampling is load-bearing. Reassigning each corrupted
    label by drawing uniformly from the group set changes the GROUP SIZES as
    dose rises -- for gas (six very unequal populations) or pg52 (2160 vs 1200)
    the top rung would end up near 50/50. Every separation statistic here scales
    with group size, so the ladder would then be measuring two things at once
    and its floor would not match the P1 null. Permutation preserves the label
    multiset exactly at every rung, so the ONLY thing varying along the ladder
    is how much true label information remains.

    Returns per-dose mean/sd of the separation over `n_rep` independent
    corruptions, plus a TREND TEST over every individual replicate.

    The trend test is a Spearman rank correlation between dose and separation,
    with a permutation p from shuffling the dose labels. It replaced a
    rung-to-rung "never rises" rule, which was the wrong instrument: with a
    handful of replicates per rung, adjacent means differ by sampling noise, so
    that rule failed on a genuinely clean ladder and would equally have passed
    a noisy flat one. Rank correlation uses every measurement instead of six
    noisy comparisons, and it does not care about the spacing of the doses.

    `dose_response` is True when rho <= -0.5 AND the permutation p < 0.05 AND
    the top rung exceeds the floor. Report rho and p, never the boolean alone.
    """
    y0 = np.asarray(labels)
    rungs = []
    for d in doses:
        seps, verds = [], []
        for _ in range(max(1, n_rep) if d > 0 else 1):
            y = y0.copy()
            if d > 0:
                hit = np.flatnonzero(rng.random(y.size) < d)
                if hit.size > 1:
                    y[hit] = y[rng.permutation(hit)]
            r = screen_populations(X, y, rng=rng, **kw)
            seps.append(r.separation)
            verds.append(r.verdict.value)
        seps = np.asarray(seps, float)
        rungs.append(dict(dose=float(d), mean=float(np.nanmean(seps)),
                          sd=float(np.nanstd(seps)), n=len(seps),
                          seps=[float(v) for v in seps], verdicts=verds))
    means = [r["mean"] for r in rungs]
    dd = np.array([r["dose"] for r in rungs for _ in r["seps"]], float)
    ss = np.array([v for r in rungs for v in r["seps"]], float)
    good = np.isfinite(ss)
    dd, ss = dd[good], ss[good]
    rho = _spearman(dd, ss) if dd.size >= 4 else float("nan")
    p = float("nan")
    if np.isfinite(rho):
        null = np.array([_spearman(rng.permutation(dd), ss) for _ in range(2000)])
        p = float((1 + np.sum(null <= rho)) / (null.size + 1))   # one-sided: we
        # predicted the sign in advance -- more corruption, less separation.
    top = means[0] if means else float("nan")
    floor = means[-1] if means else float("nan")
    ok = bool(np.isfinite(rho) and rho <= -0.5 and np.isfinite(p) and p < 0.05
              and top > floor)
    return dict(rungs=rungs, dose_response=ok, rho=float(rho), p_trend=p,
                top=top, floor=floor, doses=list(doses), n_rep=n_rep,
                n_points=int(dd.size))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties averaged. numpy only -- no scipy dependency."""
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, float)
        r[o] = np.arange(1, v.size + 1, dtype=float)
        s = v[o]
        i = 0
        while i < s.size:
            j = i
            while j + 1 < s.size and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = r[o[i:j + 1]].mean()
            i = j + 1
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / d) if d > 1e-15 else float("nan")
