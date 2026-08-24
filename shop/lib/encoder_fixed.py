"""Calibrated replacement for calculate_shannon_entropy's threshold ladder.

THE DEFECT, MEASURED
--------------------
oilgas_brain.py:159 computes fractional changes between adjacent values in a
sensor slice and buckets them with a HARDCODED ladder:

    thresholds = [-0.002, -0.0005, 0.0005, 0.002]

Those are +/-0.05% and +/-0.2% moves -- financial tick sizes. The report in
DOI 18918783 says so in print: "the exact same Shannon entropy compression
algorithm that operates on financial market data."

Applied to gas-sensor slices they saturate completely. Measured over batch 1,
all 16 sensors, 49,840 changes:

    bins [<-0.2%, -0.05%, ~0, +0.05%, >+0.2%] = [33519, 1, 4, 5, 16311]
    99.98% of all mass in the two end bins; 10 changes total in the middle three.

Consequence: per-sensor entropy takes only 2-6 DISTINCT VALUES across hundreds of
samples. It is a 3-level quantizer, not an entropy measure. That is why batch 1
reports H = 0.1431 identically for all 16 physically distinct sensors -- the
number reads the ladder's saturation, not the sensor.

THE FIX
-------
Keep the algorithm shape exactly -- same fractional changes, same 5 bins, same
normalisation. Replace only the CALIBRATION: fit the four cut points as quantiles
of the real change distribution, so the bins are occupied by construction instead
of by accident.

    thresholds = quantile(changes, [0.2, 0.4, 0.6, 0.8])

Fit once on a reference sample, then FROZEN and applied unchanged everywhere.
That keeps it a fixed encoder rather than something that re-tunes per input.

WHAT THIS DOES NOT FIX
----------------------
The second fault stands: np.diff across a sensor slice differences EIGHT
DIFFERENT QUANTITIES -- [dR, |dR|, EMAi0.001, EMAi0.01, EMAi0.1, EMAd...]. The
fractional change from a resistance delta to a moving average is not a physical
quantity. Quantile calibration makes the bins carry information about that ratio
distribution; it does not make the ratio meaningful. Fixing that means changing
what is differenced, which is a bigger change to the published encoder and is
deliberately left alone here.

Nothing in oilgas_brain.py is modified. The published path stays reproducible.
"""
import importlib.util as u
import sys

import numpy as np

OILGAS = str(__import__("pathlib").Path(__file__).resolve().parent / "oilgas_brain.py")

_spec = u.spec_from_file_location("ogb_enc", OILGAS)
_ogb = u.module_from_spec(_spec)
_spec.loader.exec_module(_ogb)

PUBLISHED_THRESHOLDS = [-0.002, -0.0005, 0.0005, 0.002]
N_BINS = 5


def slice_changes(features_128):
    """The fractional changes the published encoder actually bins, per sensor."""
    out = []
    for s in range(16):
        sd = np.asarray(features_128[s * 8:(s + 1) * 8], dtype=float)
        vals = sd[sd != 0]
        if len(vals) > 1:
            out.append(np.diff(vals) / (np.abs(vals[:-1]) + 1e-10))
    return np.concatenate(out) if out else np.array([])


def fit_thresholds(rows, qs=(0.2, 0.4, 0.6, 0.8)):
    """Quantile cut points from a reference sample. Fit once, then freeze."""
    ch = np.concatenate([slice_changes(r) for r in rows])
    ch = ch[np.isfinite(ch)]
    return list(np.quantile(ch, qs))


def entropy(values, thresholds, n_bins=N_BINS):
    """Identical to calculate_shannon_entropy except the ladder is supplied."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    changes = np.diff(values) / (np.abs(values[:-1]) + 1e-10)
    if len(changes) == 0:
        return 0.0
    bins = np.zeros(n_bins)
    for c in changes:
        if c < thresholds[0]:
            bins[0] += 1
        elif c < thresholds[1]:
            bins[1] += 1
        elif c < thresholds[2]:
            bins[2] += 1
        elif c < thresholds[3]:
            bins[3] += 1
        else:
            bins[4] += 1
    total = bins.sum()
    if total == 0:
        return 0.0
    p = bins[bins > 0] / total
    return float(-(p * np.log2(p)).sum() / np.log2(n_bins))


def process_sample(features_128, thresholds):
    """Mirror of TransponderEngine.process_sample with the calibrated ladder.

    Field names and ordering are identical so this drops into brainrow/state24
    without any other change.
    """
    f = np.asarray(features_128, dtype=float)
    ents = []
    for s in range(16):
        sd = f[s * 8:(s + 1) * 8]
        if np.any(sd != 0):
            vals = sd[sd != 0]
            ents.append(entropy(vals, thresholds) if len(vals) > 1 else 0.0)
        else:
            ents.append(0.0)
    mean_e, max_e, min_e = float(np.mean(ents)), float(np.max(ents)), float(np.min(ents))
    sig = float(np.mean(np.abs(f[f != 0]))) if np.any(f != 0) else 0.0
    return {
        "sensor_entropies": ents,
        "mean_entropy": mean_e,
        "max_entropy": max_e,
        "min_entropy": min_e,
        "entropy_spread": max_e - min_e,
        "signal_strength": sig,
        "label": None,
        "gas_name": "Unknown",
    }


# ---------------------------------------------------------------------------
# Diagnostic: does the calibrated ladder actually restore dynamic range?
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    D = "/home/voodoo/voodoo_baselines/gsad"
    GAS = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
           4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}

    X, g = [], []
    for line in open(f"{D}/batch1.dat"):
        p = line.split()
        if len(p) < 129:
            continue
        g.append(int(p[0].split(";")[0]))
        X.append([float(t.split(":")[1]) for t in p[1:129]])
    X, g = np.array(X), np.array(g)

    th = fit_thresholds(X)
    print("fitted thresholds (quantiles 0.2/0.4/0.6/0.8 of real changes):")
    print("  " + "  ".join(f"{t:+.4g}" for t in th))
    print("published ladder:")
    print("  " + "  ".join(f"{t:+.4g}" for t in PUBLISHED_THRESHOLDS))

    def occupancy(thresholds):
        tot = np.zeros(N_BINS)
        for row in X:
            for s in range(16):
                sd = row[s * 8:(s + 1) * 8]
                vals = sd[sd != 0]
                if len(vals) < 2:
                    continue
                ch = np.diff(vals) / (np.abs(vals[:-1]) + 1e-10)
                for c in ch:
                    i = (0 if c < thresholds[0] else 1 if c < thresholds[1]
                         else 2 if c < thresholds[2] else 3 if c < thresholds[3] else 4)
                    tot[i] += 1
        return tot

    for name, t in (("published", PUBLISHED_THRESHOLDS), ("calibrated", th)):
        occ = occupancy(t)
        end = (occ[0] + occ[4]) / occ.sum()
        print(f"\n{name:>10} occupancy: {occ.astype(int)}   end-bin mass {end:.4f}")

    print("\ndistinct per-sensor H values (batch 1, first 200 rows per gas):")
    print(f"  {'gas':<14}{'published':>11}{'calibrated':>12}{'cal. sd':>10}")
    for k in sorted(GAS):
        idx = np.where(g == k)[0][:200]
        if len(idx) == 0:
            continue
        pub = np.array([[entropy(X[i][s * 8:(s + 1) * 8][X[i][s * 8:(s + 1) * 8] != 0],
                                 PUBLISHED_THRESHOLDS) for s in range(16)] for i in idx])
        cal = np.array([[entropy(X[i][s * 8:(s + 1) * 8][X[i][s * 8:(s + 1) * 8] != 0],
                                 th) for s in range(16)] for i in idx])
        print(f"  {GAS[k]:<14}{len(np.unique(np.round(pub, 6))):11d}"
              f"{len(np.unique(np.round(cal, 6))):12d}{cal.std():10.4f}")

    print("\nper-sensor spread within batch 1 (do the 16 sensors differ at all?):")
    for name, t in (("published", PUBLISHED_THRESHOLDS), ("calibrated", th)):
        per = np.array([[entropy(row[s * 8:(s + 1) * 8][row[s * 8:(s + 1) * 8] != 0], t)
                         for s in range(16)] for row in X[:400]])
        m = per.mean(0)
        print(f"  {name:>10}  sensor means {m.min():.4f}..{m.max():.4f}   "
              f"spread {m.max() - m.min():.4f}")
