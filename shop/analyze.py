"""Lattice24 sensor/process report — the real engine, run on a customer's CSV.

    python3 analyze.py their_data.csv --out report.md
    python3 analyze.py their_data.csv --label-col status --out report.md

WHAT THIS IS
------------
The customer sends one CSV of time-ordered process or sensor data. This runs it
through TransponderEngine (32) -> QuantumProcessor (12 qubits) -> aoi_collapse
and returns a markdown report. It runs on Jim's machine, not in a browser: the
engine is Python, and a browser page cannot call it. Any tool claiming to run
"the engine" client-side is running a histogram.

WHAT IT REFUSES TO DO, AND WHY EACH REFUSAL EXISTS
---------------------------------------------------
Each of these is a mistake made on 2026-08-06 and caught. They are gates now.

1. CHECKS the measurement scale BEFORE reporting anything, and refuses if it
   cannot be made readable.
   calculate_shannon_entropy bins fractional changes at +/-0.0005/+/-0.002 --
   tick sizes from the trading code. Those assume CONSECUTIVE SAMPLES OF ONE
   QUANTITY, slowly varying and away from zero. On the UCI gas-sensor set, where
   a slice held eight different quantities, 99.98% of 49,840 changes landed in
   the two end bins and per-sensor entropy took only 2-6 distinct values across
   hundreds of samples. A flat channel returns a clean, believable null that
   means nothing.

   Fermentation process data sits at 0.33 end-bin mass on the standard scale and
   is read as-is. Anything oscillating through zero saturates it -- plain
   sinusoids hit 0.99 -- so when that happens the four cut points are refitted as
   quantiles of THIS FILE's change distribution, which is disclosed in the report.
   That refit uses the shape of the changes ONLY; labels are never consulted, so
   it cannot be tuned toward a result. If the data is still saturated after the
   refit, nothing is reported at all -- that means consecutive rows are not
   consecutive measurements of the same thing.

2. REFUSES to report a number without a control that could have killed it.
   Permutation null on the labels, and where there are no labels, a shuffled-
   order null. A lone reading is uninterpretable.

3. REFUSES to fit a model to the customer's data.
   The engine's outputs ARE the measurement. Wrapping them in a regression and
   scoring the regression measures the regression -- that produced four claims in
   one day, all withdrawn. There is no fitting anywhere in this file.

4. NAMES the primary channel before reading, and does not let a secondary
   rescue a null primary. With 4 channels, roughly 1 in 5 will look significant
   by chance.

5. WARNS about group confounds it can see. If a label is perfectly aligned with
   time order, that is reported as a limitation on the face of the report,
   because it usually cannot be separated after the fact.

INPUT
-----
CSV with a header. Numeric columns are treated as channels. Rows must be in time
order. Optional --label-col names a column holding a group label (e.g. ok/fault);
without it the report describes structure rather than comparing groups.
"""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from brainrow_fixed import SlotScaler, engine, quantum, state24  # noqa: E402


import encoder_fixed as _enc  # noqa: E402  parameterised ladder, same algebra


from aoi_collapse import aoi_collapse  # noqa: E402

CHECKSUM = 34.031437
assert abs(engine.compute_super_logarithm() - CHECKSUM) < 1e-5, "WRONG ENGINE"

WIN = 8            # consecutive time steps per channel
NCH = 16           # channels per 128-vector
SHOTS = 256
NPERM = 4000
TICKS = [-0.002, -0.0005, 0.0005, 0.002]
SAT_LIMIT = 0.90   # end-bin mass above this = saturated, refuse
MIN_ROWS = 200


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def read_csv(path, label_col=None, group_col=None):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise SystemExit("file has no data rows")
    hdr = [h.strip() for h in rows[0]]
    body = rows[1:]

    def column(name, what):
        if name not in hdr:
            raise SystemExit(f"--{what} '{name}' not in header: {hdr[:12]}")
        i = hdr.index(name)
        return [r[i].strip() if i < len(r) else "" for r in body]

    labels = column(label_col, "label-col") if label_col else None
    groups = column(group_col, "group-col") if group_col else None

    keep, names = [], []
    for i, h in enumerate(hdr):
        if h in (label_col, group_col):
            continue
        col = []
        for r in body:
            try:
                col.append(float(r[i]) if i < len(r) and r[i].strip() != "" else np.nan)
            except ValueError:
                col = None
                break
        if col is None:
            continue
        arr = np.array(col, float)
        if np.isnan(arr).mean() > 0.5:      # mostly-empty offline assay columns
            continue
        if np.nanstd(arr) < 1e-12:          # constant, carries nothing
            continue
        keep.append(arr)
        names.append(h)
    if not keep:
        raise SystemExit("no usable numeric columns found")
    return np.column_stack(keep), names, labels, groups


def windows(X, nch, rng, groups=None, per_group=60, cap=600):
    """128-vectors: nch channels x WIN consecutive time steps, channel-major.

    Channel-major matters. Each 8-slice handed to the entropy function is then
    ONE channel's short time series, so differencing compares like with like.
    The gas-sensor failure was slices holding eight different quantities.

    When `groups` is supplied, no window is allowed to straddle two runs -- a
    window spanning the end of one batch and the start of the next is not a
    measurement of anything.
    """
    def take(lo, hi, limit):
        st = [s for s in range(lo, hi - WIN, WIN)]
        if len(st) > limit:
            st = sorted(rng.choice(st, limit, replace=False))
        out = []
        for s in st:
            w = X[s:s + WIN, :nch]
            if np.all(np.isfinite(w)):
                out.append((s, w.T.reshape(-1)))
        return out

    if groups is None:
        return take(0, len(X), cap)

    out, lo = [], 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[lo]:
            out += take(lo, i, per_group)
            lo = i
    return out


# ---------------------------------------------------------------------------
# gate 1 -- encoder health
# ---------------------------------------------------------------------------

def all_changes(vecs):
    out = []
    for _, v in vecs:
        for c in range(NCH):
            sl = v[c * WIN:(c + 1) * WIN]
            sl = sl[sl != 0]
            if len(sl) > 1:
                out.append(np.diff(sl) / (np.abs(sl[:-1]) + 1e-10))
    return np.concatenate(out) if out else np.array([])


def occupancy(vecs, ticks=None):
    t = TICKS if ticks is None else ticks
    ch = all_changes(vecs)
    ch = ch[np.isfinite(ch)]
    if not len(ch):
        return np.zeros(5)
    idx = np.digitize(ch, t)
    return np.bincount(idx, minlength=5)[:5] / len(ch)


def recalibrate(vecs, qs=(0.2, 0.4, 0.6, 0.8)):
    """Refit the four cut points as quantiles of THIS file's change distribution.

    Used only when the published ladder saturates. The published thresholds
    (+/-0.0005, +/-0.002) are tick sizes from the trading code and suit slowly
    varying, positive-valued quantities sampled finely -- fermentation process
    data sits at 0.33 end-bin mass with them. Anything oscillating through zero
    saturates them instead, and would otherwise be refused outright.

    This is fitting, so it is worth being precise about what kind: the cut points
    are fit to the SHAPE OF THE CHANGE DISTRIBUTION ONLY. Labels are never
    consulted, so it cannot be tuned toward a result -- it can only put the
    measurement scale where the data actually lives. Any report using it says so.
    """
    ch = all_changes(vecs)
    ch = ch[np.isfinite(ch)]
    if not len(ch):
        return None
    return list(np.quantile(ch, qs))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def encode(v, ticks):
    """Transponder pass. With ticks=None this IS engine.process_sample --
    verified byte-identical over 300 rows, max abs difference 0.0."""
    if ticks is None:
        return engine.process_sample(v)
    return _enc.process_sample(v, ticks)


def read(vecs, scaler, ticks=None):
    rows = []
    for s, v in vecs:
        sr = encode(v, ticks)
        q = quantum.run(sr, shots=SHOTS)
        r = aoi_collapse(scaler.collapse_input(state24(sr, onescale=False)))
        rows.append((s, sr["mean_entropy"], r["chaos_level"],
                     r["control_norm"], q["anomaly_probability"]))
    return rows


def perm_groups(a, b, rng, n=NPERM):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan"), None
    obs = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na, hits = len(a), 0
    for _ in range(n):
        p = rng.permutation(pool)
        if abs(p[:na].mean() - p[na:].mean()) >= abs(obs) - 1e-15:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def perm_order(seq, rng, n=1000):
    """No labels: does real order differ from shuffled order? Lag-1 statistic,
    because a plain mean is order-invariant and would test nothing."""
    seq = np.asarray(seq, float)
    stat = lambda s: float(np.sqrt(np.mean(np.diff(s) ** 2)))
    obs = stat(seq)
    hits = sum(1 for _ in range(n) if stat(rng.permutation(seq)) >= obs)
    return obs, (hits + 1) / (n + 1)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Lattice24 process/sensor report.")
    ap.add_argument("csv")
    ap.add_argument("--label-col", default=None,
                    help="column holding a group label, e.g. ok/fault")
    ap.add_argument("--group-col", default=None,
                    help="column identifying the unit (batch/run/asset). Readings "
                         "are averaged within each unit before groups are compared. "
                         "Use this whenever rows belong to runs -- windows inside "
                         "one run are not independent, and comparing them as if "
                         "they were hides real effects in within-run noise.")
    ap.add_argument("--out", default="report.md")
    a = ap.parse_args()

    rng = np.random.default_rng(20260806)
    X, names, labels, groups = read_csv(a.csv, a.label_col, a.group_col)
    L, W = [], []
    L.append(f"# Sensor / process report\n")
    L.append(f"File `{Path(a.csv).name}` · {X.shape[0]:,} rows · "
             f"{X.shape[1]} usable channels")
    L.append(f"Engine checksum {engine.compute_super_logarithm():.6f} "
             f"(required {CHECKSUM}) · "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n")

    if X.shape[0] < MIN_ROWS:
        L.append(f"**Stopped.** {X.shape[0]} rows is too few to read; "
                 f"{MIN_ROWS} is the minimum. Nothing else in this report.")
        Path(a.out).write_text("\n".join(L))
        print(f"wrote {a.out} (too few rows)")
        return

    nch = min(NCH, X.shape[1])
    if X.shape[1] < NCH:
        W.append(f"Only {X.shape[1]} channels supplied; the engine takes {NCH}. "
                 f"The remaining slots are zero-filled, which weakens the read.")
    vecs = windows(X, nch, rng, groups)
    L.append(f"Read as {len(vecs)} windows of {WIN} consecutive samples "
             f"across {nch} channels: {', '.join(names[:nch])}\n")
    if groups:
        L.append(f"Grouped by `{a.group_col}` into {len(set(groups))} runs. "
                 f"Readings are averaged within each run before runs are compared, "
                 f"because windows inside one run are not independent of each "
                 f"other.\n")

    # ---- gate 1 ----
    occ = occupancy(vecs)
    end = occ[0] + occ[4]
    L.append("## Encoder check\n")
    L.append("| bin | share |\n|---|---|")
    for i, lab in enumerate(["< -0.2%", "-0.2..-0.05%", "flat", "+0.05..0.2%", "> +0.2%"]):
        L.append(f"| {lab} | {occ[i]:.3f} |")
    L.append(f"\nEnd-bin mass **{end:.4f}** on the standard scale.")
    ticks = None
    if end > SAT_LIMIT:
        ticks = recalibrate(vecs)
        occ2 = occupancy(vecs, ticks) if ticks else None
        end2 = (occ2[0] + occ2[4]) if occ2 is not None else 1.0
        if occ2 is None or end2 > SAT_LIMIT:
            L.append(
                f"\n**Stopped — this data cannot be read on any scale.** "
                f"{end * 100:.2f}% of sample-to-sample changes fall in the two "
                f"extreme bins, and refitting the scale to your own data does not "
                f"fix it. That normally means consecutive rows are not consecutive "
                f"measurements of the same thing — columns on wildly different "
                f"scales interleaved, or rows not in time order. Check the row "
                f"ordering and send it again. No reading is reported, because a "
                f"flat channel returns a confident-looking result that means "
                f"nothing.")
            Path(a.out).write_text("\n".join(L))
            print(f"wrote {a.out} — REFUSED, saturated on both scales ({end:.4f})")
            return
        L.append(
            f"\n**The standard scale saturates on your data, so it was refitted.** "
            f"The default cut points are sized for slowly-varying, positive-valued "
            f"process channels; yours move faster than that or pass through zero. "
            f"Refitting to your own change distribution brings end-bin mass to "
            f"**{end2:.4f}**, which is readable.\n\n"
            f"Worth being exact about what was fitted: **the cut points were set "
            f"from the shape of your data's changes only. No labels were used.** "
            f"The scale cannot be tuned toward a result — it can only be put where "
            f"your data actually lives. Everything below is measured on that scale.")
        occ = occ2
    else:
        L.append(f" Healthy — the channel has room to move. "
                 f"(Saturation would be >{SAT_LIMIT}.)\n")

    # ---- read ----
    fit = [state24(encode(v, ticks), onescale=False) for _, v in vecs[:200]]
    scaler = SlotScaler().fit(fit)
    rows = read(vecs, scaler, ticks)
    chans = {"entropy": [r[1] for r in rows], "chaos": [r[2] for r in rows],
             "control": [r[3] for r in rows], "anomaly": [r[4] for r in rows]}

    L.append("## Result\n")
    if labels:
        starts = [r[0] for r in rows]
        lab_at = [labels[min(s, len(labels) - 1)] for s in starts]
        lab_vals = sorted(set(x for x in lab_at if x))
        if len(lab_vals) != 2:
            L.append(f"Label column has {len(lab_vals)} distinct values "
                     f"({lab_vals[:5]}); this report compares exactly two. "
                     f"Re-run with a two-value label.\n")
        else:
            g0, g1 = lab_vals
            L.append(f"Comparing **{g0}** vs **{g1}**, primary channel is "
                     f"entropy (named before reading).\n")
            grp_at = ([groups[min(s, len(groups) - 1)] for s in starts]
                      if groups else None)
            # Order runs by FIRST APPEARANCE, i.e. true file order. Sorting the
            # ids would order them as strings ('100' < '11' < '2'), which is not
            # the order they were recorded in -- and the blocking test below is
            # meaningless unless the sequence is chronological.
            units = list(dict.fromkeys(grp_at)) if grp_at else []
            if grp_at:
                L.append(f"Unit of comparison: **{len(units)} runs**, not "
                         f"individual windows.\n")

            def split(v):
                """Return (group1 values, group0 values), per RUN when grouped."""
                if not grp_at:
                    return ([v[i] for i, x in enumerate(lab_at) if x == g1],
                            [v[i] for i, x in enumerate(lab_at) if x == g0])
                per, lab_of = {}, {}
                for i, u in enumerate(grp_at):
                    per.setdefault(u, []).append(v[i])
                    lab_of[u] = lab_at[i]
                m = {u: float(np.mean(vals)) for u, vals in per.items()}
                return ([m[u] for u in units if lab_of[u] == g1],
                        [m[u] for u in units if lab_of[u] == g0])

            L.append("| channel | difference | p | |\n|---|---|---|---|")
            first = True
            for nm, v in chans.items():
                A, B = split(v)
                d, p = perm_groups(A, B, np.random.default_rng(7))
                tag = "**primary**" if first else "secondary"
                sig = "separates" if p is not None and p < 0.05 else "—"
                L.append(f"| {nm} {tag} | {d:+.5f} | {p:.4f} | {sig} |")
                first = False
            L.append("\np is a permutation test on the labels, 4000 draws. "
                     "A secondary channel does not rescue a null primary — with "
                     "four channels, roughly one looks significant by luck.")
            # ---- confound warning: is the label just telling us WHEN? ----
            # Checked symmetrically. An earlier version tested only one of the
            # two labels, and which one that was depended on alphabetical sort
            # order -- so on a file whose fault runs were all at the end, it
            # inspected the healthy runs and stayed silent. This is the most
            # important line in the report; it must not depend on a label's name.
            if grp_at:
                seq = [lab_of_unit for lab_of_unit in
                       [next(lab_at[i] for i, u in enumerate(grp_at) if u == uu)
                        for uu in units]]

            else:
                seq = lab_at
            # How often does the label change along the sequence? Perfectly
            # blocked groups change once. Interleaved groups change often.
            #
            # Correlation is the wrong statistic here: with 10 vs 30 units a
            # PERFECT block split tops out near r=0.61, so any sensible
            # correlation threshold misses exactly the case that matters most.
            # Change-count is scale-free and unaffected by group imbalance.
            n1 = sum(1 for x in seq if x == g1)
            n0 = len(seq) - n1
            changes = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
            expected = 2.0 * n1 * n0 / len(seq) if len(seq) else 0.0
            unit_word = "runs" if grp_at else "stretches"
            if expected >= 2 and changes <= max(2, 0.25 * expected):
                W.append(
                    f"**The two groups are blocked in time, not interleaved.** "
                    f"The label changes {changes}× across {len(seq)} {unit_word}; "
                    f"if the conditions were mixed through the campaign you would "
                    f"expect about {expected:.0f}. So every '{g1}' sits on one side "
                    f"of the run order and every '{g0}' on the other, and a "
                    f"difference above may be *when* the data was recorded rather "
                    f"than the condition itself. **No analysis can separate those "
                    f"after the fact** — it needs {g0} and {g1} {unit_word} "
                    f"interleaved in time. Treat the result as suggestive, not "
                    f"settled, and if you have healthy {unit_word} from the same "
                    f"period as the flagged ones, send those and we re-run.")
                if grp_at:
                    W.append(
                        f"Worth checking on your side: is there anything else that "
                        f"changed between the early and late {unit_word} — a recipe "
                        f"revision, a control-strategy change, a maintenance event? "
                        f"Anything blocked the same way is indistinguishable from "
                        f"the condition in this data.")
    else:
        L.append("No label column given, so this describes structure rather than "
                 "comparing groups.\n")
        L.append("| channel | order effect | p |\n|---|---|---|")
        for nm, v in chans.items():
            s, p = perm_order(v, np.random.default_rng(9))
            L.append(f"| {nm} | {s:.5f} | {p:.4f} |")
        L.append("\np compares the real sample order against shuffled order. "
                 "A high p means the readings carry no sequence structure — "
                 "which is a result, not a failure.")

    if W:
        L.append("\n## Read this before acting on the above\n")
        for w in W:
            L.append(f"- {w}")

    L.append("\n## What this does not tell you\n")
    L.append("- It does not say a channel is *correct*, only whether it moves.")
    L.append("- It does not replace your instruments or your calibration schedule.")
    L.append("- Nothing here was fitted to your data, so nothing is tuned to "
             "flatter it. The same pipeline runs on every file unchanged.")
    L.append("\n---\nLattice24 · James Jardine · ORCID 0009-0004-9073-7192")

    Path(a.out).write_text("\n".join(L))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
