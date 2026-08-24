#!/usr/bin/env python3
"""
Order-invariance annex for customer reports -- the two engine-free arms.

Runs every numeric channel of a customer CSV through the merged QBH screen
(harness.screen_window), which ALWAYS executes:
    own-shuffle null, scale-matched null, sorted-surrogate null,
    ARM 5: IAAFT surrogate null   (distribution AND spectrum preserved)
    ARM 6: permutation entropy    (orders 3/4, blind to amplitude)
plus the engine-free baselines (lag-1 autocorr, OLS trend).

Interpretation rules come from ~/qbh/ARMS_5_6.md and are binding:
  - INVARIANT is a RESULT, not a failure.
  - Arm 6 SILENCE at n=24 is weak evidence of absence; an arm-6 FIRE is meaningful.
  - lag-1 reading sequence does not mean the collapse engine does; the bench
    measured READS 0/280 on temporal domains while lag-1 fired 211/280.
"""
import sys, csv, statistics
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "lib"))
import numpy as np
from harness import screen_window

WINDOWS_PER_CHANNEL = 3


def numeric_columns(path):
    rows = list(csv.reader(open(path, errors="replace")))
    if len(rows) < 30:
        return []
    header = [h.strip() for h in rows[0]]
    cols = [[] for _ in header] if header else []
    ncol = len(rows[1])
    cols = [[] for _ in range(ncol)]
    names = header[:ncol] if header and len(header) == ncol else [f"col{i}" for i in range(ncol)]
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
    return out[:4]


def zof(d, k):
    v = d.get(k)
    return getattr(v, "z", None)


def annex(path):
    chans = numeric_columns(path)
    if not chans:
        return "\n## Order-invariance annex\n\nNot enough numeric data to screen.\n"
    lines = ["\n## Order-invariance annex (engine-free arms)\n",
             "Every channel was screened with the two surrogate arms wired into this",
             "shop's engine (IAAFT spectral null + permutation entropy), plus plain",
             "lag-1 autocorrelation as the engine-free baseline.\n",
             "| channel | windows | INVARIANT | IAAFT med z | pe3 fires | lag-1 fires |",
             "|---|---|---|---|---|---|"]
    for name, series in chans:
        x = np.asarray(series, float)
        idx = np.linspace(200, max(200, len(x) - 24), WINDOWS_PER_CHANNEL).astype(int)
        inv, zs, pe_f, l1_f = 0, [], 0, 0
        for st in idx:
            w = x[st - 24:st]
            if len(w) < 24 or np.std(w) == 0:
                continue
            r = screen_window(list(map(float, w)), rng=np.random.default_rng(st))
            if str(r.verdict).endswith("INVARIANT"):
                inv += 1
            zs.append(zof(r.arms, "iaaft") or 0.0)
            pe = r.baseline.get("pe3") or r.arms.get("pe3_shuf")
            if pe is not None and abs(getattr(pe, "z", 0)) >= 2.5:
                pe_f += 1
            l1 = r.baseline.get("lag1")
            if l1 is not None and abs(l1.z) >= 2.5:
                l1_f += 1
        if zs:
            lines.append(f"| {name} | {len(zs)} | {inv}/{len(zs)} | "
                         f"{statistics.median(zs):+.2f} | {pe_f}/{len(zs)} | {l1_f}/{len(zs)} |")
    lines += [
        "\nReading this table:",
        "- **INVARIANT** means the ordering carries nothing beyond the values themselves.",
        "  That is information, not a failure.",
        "- A high **lag-1** rate with all-invariant windows is expected: simple persistence",
        "  is real structure, but it is not what the collapse engine measures.",
        "- The permutation-entropy arm is deliberately cautious at 24 points; its silence",
        "  is weak evidence, its fire is strong.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(annex(sys.argv[1]))
