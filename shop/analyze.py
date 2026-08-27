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

import encoder_gate as _gate            # noqa: E402  per-channel encoder health
import order_screen as _order           # noqa: E402  the arm that replaced perm_order
import value_note as _value            # noqa: E402  the dollar figure, or the refusal

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


# perm_order was REMOVED on 2026-08-26. It returned the upper tail of an
# RMS-successive-difference statistic against a plain shuffle. Real ordered data
# is smoother than its own shuffle, so the observed value sat in the LOWER tail
# and the printed p went to 1.0 -- which the report described as "no sequence
# structure". A pure ramp scored 1.0000 and read as structureless; the same file
# with every row scrambled scored 0.0250 and read as MORE structured than the
# correct one. Flipping the tail was tested and is worse: flipped, a straight
# line fires at 0.0010. Do not reinstate either version. The replacement is
# order_screen.screen_channel, which a straight line cannot pass because a
# monotone window has one ordinal pattern. Full measurement:
# ~/BUILDS/SHOP_TEST_2026-08-26.md and ~/BUILDS/WORKFLOW_VERIFY_2026-08-26.md.


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
    # Pricing. Every one of these comes from the customer. There is no default
    # price, no default emission factor and no default period anywhere in this
    # program: if any is missing the report prints no figure and says what is
    # missing. The price per tonne is NEVER hardcoded -- it is a market rate, it
    # moves, and it travels with its source, its date and its market.
    # See value_note.py.
    ap.add_argument("--value-col", default=None,
                    help="column that meters the quantity being lost")
    ap.add_argument("--value-unit", default=None,
                    help="what one unit of that column is, e.g. 'Mscf per hour'")
    ap.add_argument("--hours", type=float, default=None,
                    help="hours of process time this record covers")
    ap.add_argument("--co2e-per-unit", type=float, default=None,
                    help="tonnes CO2e per one unit of that column (YOUR factor)")
    ap.add_argument("--co2e-source", default=None,
                    help="where that emission factor came from")
    ap.add_argument("--price-per-tonne", type=float, default=None,
                    help="market price per tonne CO2e AT THE TIME OF PURCHASE")
    ap.add_argument("--price-source", default=None,
                    help="where that price came from")
    ap.add_argument("--price-date", default=None,
                    help="the date that price was taken")
    ap.add_argument("--market", default=None,
                    help="which market: avoidance/removal, compliance/voluntary")
    ap.add_argument("--currency", default="USD")
    a = ap.parse_args()

    rng = np.random.default_rng(20260806)
    X, names, labels, groups = read_csv(a.csv, a.label_col, a.group_col)
    L, W = [], []
    L.append(f"# Sensor / process report\n")
    L.append(f"File `{Path(a.csv).name}` · {X.shape[0]:,} rows · "
             f"{X.shape[1]} usable channels")
    L.append(f"Engine build {engine.compute_super_logarithm():.6f} · "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    L.append("*That number identifies which transponder configuration ran. It "
             "takes no data and returns the same value on the right input, the "
             "wrong input, or none — it is a build stamp, not a check on your "
             "file. The checks on your file are below, and each one can fail.*\n")

    if X.shape[0] < MIN_ROWS:
        L.append(f"**Stopped.** {X.shape[0]} rows is too few to read; "
                 f"{MIN_ROWS} is the minimum. Nothing else in this report.")
        Path(a.out).write_text("\n".join(L))
        print(f"wrote {a.out} (too few rows)")
        return

    nch = min(NCH, X.shape[1])
    if X.shape[1] < NCH:
        W.append(f"{X.shape[1]} usable channels were read; the engine reads "
                 f"{NCH} slots, so {NCH - X.shape[1]} sat empty and contribute "
                 f"nothing. If you sent more than {X.shape[1]} numeric columns, "
                 f"the missing ones held a value that would not parse as a "
                 f"number (a blank, an `N/A`, a units suffix) — one such cell "
                 f"drops the whole column. Nothing was misaligned; the columns "
                 f"that were read were read correctly.")
    vecs = windows(X, nch, rng, groups)
    if groups is not None and len(vecs) < 3:
        L.append(f"\n**Stopped — grouping by `{a.group_col}` leaves nothing to "
                 f"read.** A window is {WIN} consecutive rows of one run, and no "
                 f"window may straddle two runs, so a run shorter than {WIN} "
                 f"rows contributes none. Your file has "
                 f"{len(set(groups)):,} runs across {X.shape[0]:,} rows and "
                 f"produced {len(vecs)} usable window(s).")
        L.append(f"\nThis is a shape problem in the request, not a finding "
                 f"about your process. Either `{a.group_col}` is not the column "
                 f"that identifies a run — a row id or a timestamp will do this "
                 f"— or the runs are genuinely shorter than {WIN} samples, in "
                 f"which case send the file without `--group-col` and it will "
                 f"be read as one continuous record.")
        Path(a.out).write_text("\n".join(L))
        print(f"wrote {a.out} — REFUSED, grouping leaves {len(vecs)} windows")
        return

    available = len(range(0, len(X) - WIN, WIN))
    used_rows = len(vecs) * WIN
    L.append(f"Read as {len(vecs)} windows of {WIN} consecutive samples "
             f"across {nch} channels: {', '.join(names[:nch])}")
    if len(vecs) < available:
        L.append(f"\nThat is a random sample of {len(vecs)} of the "
                 f"{available:,} non-overlapping windows your file contains — "
                 f"**{used_rows:,} of {X.shape[0]:,} rows, "
                 f"{100.0 * used_rows / X.shape[0]:.1f}%.** The sample is drawn "
                 f"once with a fixed seed, so re-running this file gives the "
                 f"same windows and the same report.\n")
    else:
        L.append(f"\nThat is every non-overlapping window in the file — "
                 f"{used_rows:,} of {X.shape[0]:,} rows, "
                 f"{100.0 * used_rows / X.shape[0]:.1f}%.\n")
    if groups:
        L.append(f"Grouped by `{a.group_col}` into {len(set(groups))} runs. "
                 f"Readings are averaged within each run before runs are compared, "
                 f"because windows inside one run are not independent of each "
                 f"other.\n")

    # ---- gate 1 -- PER CHANNEL, with a refusal that can fire ----
    #
    # Pooled was the defect. Measured 2026-08-26 on the 16-channel refinery
    # file: pooled end-bin mass 0.5407 printed as "Healthy", while sensor_01
    # sat at 0.9726 and sensor_02 at 0.9864, both past the gate. See
    # encoder_gate.py for why the old refusal branch was unreachable.
    gates = _gate.gate_all(vecs, names, nch)
    L.append("## Encoder check — every channel, separately\n")
    L.append("The measurement ladder is checked on each channel on its own. An "
             "earlier version of this report pooled them, which let two "
             "saturated channels hide behind fourteen healthy ones and printed "
             "\"Healthy\" over the top.\n")
    L.append("| channel | end-bin mass | after refit | status |\n|---|---|---|---|")
    for g in gates:
        refit = "—" if g["refit_end"] is None else f"{g['refit_end']:.4f}"
        end = "—" if g["end"] != g["end"] else f"{g['end']:.4f}"
        L.append(f"| {g['channel']} | {end} | {refit} | {g['status']} |")
    refused = [g for g in gates if g["status"] == "REFUSED"]
    readable = [g for g in gates if g["status"] in ("readable", "refitted")]
    refitted = [g for g in gates if g["status"] == "refitted"]
    L.append(f"\nEnd-bin mass is the share of sample-to-sample changes landing "
             f"in the two extreme bins. Above {SAT_LIMIT} the ladder is "
             f"saturated and the channel is re-checked on a ladder refit to its "
             f"own changes; if no ladder can be placed on it, it is refused and "
             f"nothing is reported for it.")

    if not readable:
        L.append(f"\n**Stopped — this data cannot be read on any scale.** "
                 f"All {len(gates)} channels were refused:\n")
        for g in refused:
            L.append(f"- `{g['channel']}` — {g['why']}")
        L.append("\nNo reading is reported, because a flat or collapsed channel "
                 "returns a confident-looking result that means nothing. The "
                 "usual causes are columns on wildly different scales "
                 "interleaved into one file, a channel logged at a resolution "
                 "coarser than the movement you care about, or rows that are "
                 "not consecutive measurements of the same quantity.")
        Path(a.out).write_text("\n".join(L))
        print(f"wrote {a.out} — REFUSED, no channel readable")
        return

    if refused:
        W.append("**" + str(len(refused)) + " of " + str(len(gates)) +
                 " channels were refused and are not in any number below**: " +
                 "; ".join(f"`{g['channel']}` — {g['why']}" for g in refused))

    # The reading ladder is one scale for the whole 128-vector, so it is
    # refitted when ANY readable channel needed it. Which channels needed it is
    # on the table above rather than hidden behind a pooled figure.
    ticks = None
    if refitted:
        ticks = recalibrate(vecs)
        L.append(f"\n**{len(refitted)} of {len(gates)} channels saturate the "
                 f"standard ladder, so the reading scale was refit** to the "
                 f"shape of this file's own changes. Labels were never "
                 f"consulted, so the scale cannot be tuned toward a result — "
                 f"it can only be put where your data lives. One ladder is used "
                 f"for the whole read, so it is refit whenever any readable "
                 f"channel needs it; the table above says which ones did.")
    # ---- read ----
    fit = [state24(encode(v, ticks), onescale=False) for _, v in vecs[:200]]
    scaler = SlotScaler().fit(fit)
    rows = read(vecs, scaler, ticks)
    chans = {"entropy": [r[1] for r in rows], "chaos": [r[2] for r in rows],
             "control": [r[3] for r in rows], "anomaly": [r[4] for r in rows]}

    L.append("## Result\n")
    _order_results = []
    _label_result = None
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
                # perm_groups returns p=None when a side has fewer than three
                # values -- one run, or every run a singleton. Formatting None
                # with :.4f raised TypeError and killed the whole report, so a
                # customer with one batch id got a Python traceback instead of a
                # report. Found 2026-08-26 testing --group-col with pricing.
                pcell = "not enough runs" if p is None else f"{p:.4f}"
                dcell = "—" if d != d else f"{d:+.5f}"
                L.append(f"| {nm} {tag} | {dcell} | {pcell} | {sig} |")
                if first and p is None:
                    W.append(
                        f"**The groups could not be compared.** Splitting by "
                        f"`{a.group_col}` left fewer than three runs on one "
                        f"side, so the permutation test has nothing to permute. "
                        f"That is a shape problem in the request, not a finding "
                        f"about your process — it needs several runs of each "
                        f"condition.")
                if first:
                    # The primary verdict travels to the money block. Without
                    # this the priced section was byte-identical for a file
                    # whose labels were coin flips and one at p = 0.0002.
                    _label_result = ({"channel": nm, "p": p, "g0": g0, "g1": g1,
                                      "separates": bool(p < 0.05)}
                                     if p is not None else None)
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
        L.append("### Does the order of your rows carry information?\n")
        # perm_order used to answer this and answered it backwards -- see
        # order_screen.py. It is replaced by permutation entropy against a
        # shuffle of the same values, which a straight line cannot pass because
        # a monotone window realises one ordinal pattern and is refused
        # structurally, not by a threshold.
        keep = [i for i, g in enumerate(gates) if g["status"] != "REFUSED"]
        res = [(names[i], _order.screen_channel(X[:, i],
                                                np.random.default_rng(4242 + i)))
               for i in keep]
        _order_results = res
        L.append("| channel | windows read | ordinal structure | median z | "
                 "straight-line windows | trend / persistence |")
        L.append("|---|---|---|---|---|---|")
        for nm, r in res:
            mz = "—" if r["median_z"] != r["median_z"] else f"{r['median_z']:+.2f}"
            L.append(f"| {nm} | {r['ran']}/{r['windows']} | "
                     f"{r['fired']}/{r['ran']} | {mz} | {r['degenerate']} | "
                     f"{r['trend']}/{r['windows']} |")
        n_reads = sum(1 for _, r in res if r["reads"])
        touched = max((r["rows_touched"] for _, r in res), default=0)
        L.append(f"\n**{n_reads} of {len(res)} channels carry ordinal structure "
                 f"beyond their own values.**\n")
        L.append(f"Coverage, exactly: {len(res)} of {X.shape[1]} channels "
                 f"screened, {_order.n_windows(X.shape[0])} windows of "
                 f"{_order.PTS} consecutive rows plus one window of "
                 f"{_order.PTS} points sampled across the whole record, per "
                 f"channel — **{touched:,} of {X.shape[0]:,} rows, "
                 f"{100.0 * touched / X.shape[0]:.1f}% of your file.**\n")
        L.append("How to read the columns:\n")
        L.append(f"- **ordinal structure** counts the windows where the real "
                 f"order of the 24 points differs from {_order.NSHUF} shuffles "
                 f"of the same 24 values, at |z| ≥ {_order.WIN_Z}. A channel is "
                 f"only called a read when a majority of its windows fire and "
                 f"at least {_order.MIN_FIRES} do. On pure iid noise single "
                 f"windows fire about 5% of the time, which is why one window "
                 f"is not an answer.")
        L.append("- **straight-line windows** are windows the test refused to "
                 "run because the 24 points were monotone. A straight line has "
                 "one ordinal pattern, so this statistic is 0 by construction "
                 "and cannot fire on it — that is a structural refusal, not a "
                 "silent pass.")
        L.append("- **trend / persistence** is plain lag-1 autocorrelation "
                 "against the same shuffles. It is reported separately and it "
                 "is never the headline: **a straight line passes it with the "
                 "strongest possible score.** A high count here with no ordinal "
                 "structure means your channel drifts, which is real and is not "
                 "evidence of anything beyond a drift.")
        L.append(f"- **Multiplicity.** {len(res)} channels were tested. At the "
                 f"single-window level about 5% of windows fire on pure noise, "
                 f"so roughly {0.05 * len(res) * _order.n_windows(X.shape[0]):.0f} "
                 f"stray window-fires are expected across this file by luck "
                 f"alone. The majority rule is what keeps those out of the "
                 f"channel count: measured on 16 channels of pure noise, it "
                 f"returned 0 channels. That count, not the per-window column, "
                 f"is the number to read.")
        near = [(nm, r) for nm, r in res
                if not r["reads"] and r["ran"]
                and r["fired"] >= max(_order.MIN_FIRES, (r["ran"] + 1) // 2) - 2]
        if near:
            L.append("- **Channels close to the line, named so you are not "
                     "surprised by them later.** These did not clear the "
                     "majority rule, and a different random seed could move one "
                     "of them across it: "
                     + ", ".join(f"`{nm}` ({r['fired']}/{r['ran']})"
                                 for nm, r in near)
                     + ". The count above is reproducible — the seed is fixed — "
                       "but reproducible is not the same as stable, and a "
                       "borderline channel is a borderline channel.")
        L.append("\nWhat this arm is **not**: it is not the collapse engine. "
                 "The collapse engine's own verdict on the same windows is in "
                 "the annex at the end of this report, and the two answer "
                 "different questions. Neither stands in for the other.")

    if W:
        L.append("\n## Read this before acting on the above\n")
        for w in W:
            L.append(f"- {w}")

    priced_channel = None
    if a.value_col:
        priced_channel = dict(_order_results).get(a.value_col)
    L += _value.value_block(X, names, a, priced_channel, _label_result)

    L.append("\n## What this does not tell you\n")
    L.append("- It does not say a channel is *correct*, only whether it moves.")
    L.append("- It does not replace your instruments or your calibration schedule.")
    L.append("- Two things here ARE fitted to your data, and both are "
             "unsupervised: the slot scaler is fit to the first 200 windows of "
             "your file, and the measurement ladder is refit to your own change "
             "distribution when the standard one saturates. Neither ever sees a "
             "label, so neither can be tuned toward a result — but it is not "
             "true that nothing was fitted, and an earlier version of this "
             "report said so.")
    L.append("- No model is fitted to predict anything. The engine's outputs "
             "are the measurement; they are not fed to a regression and scored.")
    L.append("\n---\nLattice24 · James Jardine · ORCID 0009-0004-9073-7192")

    Path(a.out).write_text("\n".join(L))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
