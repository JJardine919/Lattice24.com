#!/usr/bin/env python3
"""Collapse-engine order verdict, per channel -- the annex at the end of a report.

Every channel is screened through the merged QBH harness (`harness.screen_window`),
which always runs, in one call:
    C0 flat-degenerate instrument gate, C1 own-shuffle, C2 scale-matched,
    C3 sorted surrogate, C5 IAAFT (distribution AND spectrum preserved),
    C6 permutation entropy against both pools,
plus the engine-free baselines (lag-1 autocorrelation, OLS trend).

WHAT WAS WRONG HERE BEFORE 2026-08-26
-------------------------------------
1. "Every channel was screened" meant `numeric_columns()[:4]` -- 4 of 16
   channels, 3 windows of 24 rows each, 0.4% of the file. The sentence was
   false and the real numbers appeared nowhere.
2. Only INVARIANT was counted and only INVARIANT was explained. A window that
   came back DRIFT ("reads, but its own sorted order reads as well or better")
   was counted merely as "not INVARIANT", so "2/3 INVARIANT" read to a customer
   as if the third window had found something. DRIFT is a REJECTION. Every
   verdict is now printed by name.
3. The three explanatory bullets were a hardcoded string printed
   unconditionally -- including on a file where lag-1 fired 0/3, and on a
   150-row job where the table was empty. Same defect class as the p-value
   prose it sat two inches below. The notes are now conditional on what the
   run actually produced.
4. `IAAFT med z` was a bare sigma with no percentile and the fire columns were
   counts against an undisclosed threshold. Both are now stated.
"""
import csv
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from harness import screen_window, instrument_gate  # noqa: E402

WINDOWS_PER_CHANNEL = 3
PTS = 24
#: |z| a baseline arm must reach to count as a fire. Stated in the report.
FIRE_Z = 2.5
#: Hard ceiling on channels screened, so a 200-column file cannot make the
#: report run for an hour. Whatever the number ends up being, it is printed.
MAX_CHANNELS = 16


def numeric_columns(path):
    rows = list(csv.reader(open(path, errors="replace")))
    if len(rows) < 30:
        return [], 0
    header = [h.strip() for h in rows[0]]
    ncol = len(rows[1])
    cols = [[] for _ in range(ncol)]
    names = (header[:ncol] if header and len(header) == ncol
             else [f"col{i}" for i in range(ncol)])
    for r in rows[1:]:
        if len(r) != ncol:
            continue
        for i, v in enumerate(r):
            try:
                cols[i].append(float(v))
            except ValueError:
                pass
    out = [(names[i], c) for i, c in enumerate(cols) if len(c) >= 60]
    out.sort(key=lambda kv: -len(kv[1]))
    return out[:MAX_CHANNELS], len(out)


def zof(d, k):
    return getattr(d.get(k), "z", None)


def annex(path):
    chans, total = numeric_columns(path)
    if not chans:
        return ("\n## Order-invariance annex (collapse engine)\n\n"
                "Not enough numeric data to screen — no channel had the "
                f"{PTS * WINDOWS_PER_CHANNEL} usable rows this section needs. "
                "Nothing is reported here rather than a table with no rows in "
                "it.\n")

    gate_ok, gate_msg = instrument_gate()
    n_rows = max(len(c) for _, c in chans)

    lines = ["\n## Order-invariance annex (collapse engine)\n"]
    if not gate_ok:
        lines.append("**The instrument gate failed** — a constant vector did "
                     "not read chaos 0, so this harness is failing open and "
                     "nothing in this annex may be quoted. Verbatim: "
                     f"`{gate_msg}`\n")
        return "\n".join(lines)

    verdicts, rows, all_iaaft = {}, [], []
    # DISTINCT row indices, not a running total. Every channel screens the SAME
    # three window positions (they depend only on record length), so adding PTS
    # per channel counted the same 72 rows sixteen times: it printed "1,152 of
    # 17,518 rows, 6.6%" where the truth is 72 rows and 0.41%, and on a legal
    # 300-row upload it printed "1,152 of 300 rows, 384.0% of the file" -- under
    # the word "exactly", in the section rewritten to end a false coverage
    # claim. Found by DooDoo 2026-08-26 and reproduced before fixing.
    touched = set()
    for name, series in chans:
        x = np.asarray(series, float)
        idx = np.linspace(PTS, max(PTS, len(x)), WINDOWS_PER_CHANNEL).astype(int)
        per = {}
        zs, pe_f, l1_f, ran = [], 0, 0, 0
        for st in idx:
            w = x[max(0, st - PTS):st]
            if len(w) < PTS or np.std(w) == 0:
                continue
            r = screen_window(list(map(float, w)),
                              rng=np.random.default_rng(int(st)),
                              gate_checked=True)
            v = str(r.verdict.value)
            per[v] = per.get(v, 0) + 1
            verdicts[v] = verdicts.get(v, 0) + 1
            ran += 1
            touched.update(range(max(0, int(st) - PTS), int(st)))
            z = zof(r.arms, "iaaft")
            if z is not None and np.isfinite(z):
                zs.append(float(z))
                all_iaaft.append(float(z))
            pe = r.baseline.get("pe3")
            if pe is not None and pe.ran and abs(pe.z) >= FIRE_Z:
                pe_f += 1
            l1 = r.baseline.get("lag1")
            if l1 is not None and l1.ran and abs(l1.z) >= FIRE_Z:
                l1_f += 1
        if ran:
            rows.append((name, ran, per, statistics.median(zs) if zs else None,
                         pe_f, l1_f))

    if not rows:
        return ("\n## Order-invariance annex (collapse engine)\n\n"
                "No window in this file could be screened — every candidate "
                "window was constant or too short. Nothing is reported here.\n")

    lines.append(
        f"**Coverage, exactly: {len(rows)} of {total} numeric channels, "
        f"{WINDOWS_PER_CHANNEL} windows of {PTS} consecutive rows each. Every "
        f"channel is read at the SAME row positions, so the rows this section "
        f"actually looked at are {len(touched):,} of {n_rows:,} — "
        f"{100.0 * len(touched) / max(1, n_rows):.2f}% of the file.** It is a "
        f"spot check on the collapse engine's own order verdict, not a sweep of "
        f"your record.\n")
    lines.append(f"Instrument gate: `{gate_msg}`\n")
    lines.append("| channel | windows | verdicts | IAAFT z (median) | "
                 f"pe3 |z|≥{FIRE_Z} | lag-1 |z|≥{FIRE_Z} |")
    lines.append("|---|---|---|---|---|---|")
    for name, ran, per, medz, pe_f, l1_f in rows:
        vs = ", ".join(f"{k} {v}" for k, v in sorted(per.items()))
        mz = "—" if medz is None else f"{medz:+.2f}"
        lines.append(f"| {name} | {ran} | {vs} | {mz} | {pe_f}/{ran} | {l1_f}/{ran} |")

    lines.append("\n**What each verdict means. Only READS is a find; every "
                 "other verdict is the screen refusing.**\n")
    meanings = {
        "READS": "survived every control — the ordering carries something the "
                 "value distribution, the power spectrum and scale-matched "
                 "noise do not account for.",
        "INVARIANT": "the ordering does nothing beyond the values themselves. "
                     "That is a result, not a failure.",
        "DRIFT": "**a rejection.** The window reads, but its own values sorted "
                 "into monotone order read as well or better — so what was "
                 "read is a trend, not sequence content.",
        "SPECTRAL": "**a rejection.** The window reads, but so does every "
                    "surrogate holding its exact value distribution and its "
                    "power spectrum. A linear process with the same "
                    "second-order structure reproduces it.",
        "NOISE_LEVEL": "**a rejection.** Not separated from Gaussian noise "
                       "matched to the same mean and spread.",
        "FLAT": "the window was constant. There is no direction to read.",
        "VOID": "**quote nothing from this window.** The instrument gate did "
                "not pass for it.",
    }
    for v in sorted(verdicts):
        lines.append(f"- **{v}** ({verdicts[v]} window"
                     f"{'' if verdicts[v] == 1 else 's'}) — "
                     f"{meanings.get(v, 'unrecognised verdict.')}")

    n_reads = verdicts.get("READS", 0)
    total_win = sum(verdicts.values())
    lines.append(f"\n**{n_reads} of {total_win} screened windows returned "
                 f"READS.**")
    if n_reads == 0:
        lines.append("So on the windows it saw, the collapse engine found "
                     "nothing in the ORDER of your rows that survives its own "
                     "controls. Read that as a property of this screen as much "
                     "as of your data — see the note on lag-1 below if it "
                     "fired.")

    # These notes used to print unconditionally, including where they were
    # flatly contradicted by the table above them. Each one now requires the
    # thing it describes to have actually happened.
    if all_iaaft:
        lo = min(all_iaaft)
        hi = max(all_iaaft)
        lines.append(f"\n- IAAFT z ran from {lo:+.2f} to {hi:+.2f} across "
                     f"{len(all_iaaft)} windows; the fire threshold is "
                     f"|z| ≥ {FIRE_Z}, and the IAAFT pool is 200 surrogates, so "
                     f"no p below 1/201 ≈ 0.005 is expressible by this arm "
                     f"whatever the z says.")
    lag_fires = sum(r[5] for r in rows)
    lag_total = sum(r[1] for r in rows)
    if lag_fires and n_reads == 0:
        lines.append(f"- Plain lag-1 autocorrelation fired on {lag_fires} of "
                     f"{lag_total} windows while the collapse engine returned "
                     f"READS on none. Simple persistence is real structure and "
                     f"it is in your data; it is not what the collapse engine "
                     f"measures. When those two disagree this way, the null "
                     f"belongs to the instrument, not to your file.")
    elif lag_fires == 0:
        lines.append(f"- Plain lag-1 autocorrelation fired on 0 of {lag_total} "
                     f"windows. There is no simple persistence here either — "
                     f"successive rows are not predicting each other at all, "
                     f"which is what scrambled or independently sampled rows "
                     f"look like.")
    pe_fires = sum(r[4] for r in rows)
    if pe_fires:
        lines.append(f"- The permutation-entropy arm fired on {pe_fires} of "
                     f"{lag_total} windows. At {PTS} points its silence is weak "
                     f"evidence of absence; a fire is meaningful.")
    else:
        lines.append(f"- The permutation-entropy arm fired on 0 of {lag_total} "
                     f"windows. At {PTS} points that arm is deliberately "
                     f"cautious, so its silence here is weak evidence, not a "
                     f"finding.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(annex(sys.argv[1]))
