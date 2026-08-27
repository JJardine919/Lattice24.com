"""Encoder health, PER CHANNEL, with a refusal branch that can actually fire.

WHY PER CHANNEL
---------------
`analyze.occupancy` pooled sample-to-sample changes across all 16 channels
before applying SAT_LIMIT. Measured 2026-08-26 on the 16-channel refinery test
file: the pooled figure was 0.5407 and the report printed "Healthy -- the
channel has room to move", while sensor_01 sat at 0.9726 and sensor_02 at
0.9864, both past the gate, and sensor_09 at 0.8911, 0.009 short of it. A
pooled statistic is structurally incapable of seeing that. analyze.py's own
docstring exists to prevent exactly this failure -- "a flat channel returns a
clean, believable null that means nothing" -- and it was live in the report.

WHY THE OLD REFUSAL COULD NOT FIRE
----------------------------------
The refusal read: saturated on the standard ladder, refit, and if STILL
saturated, report nothing. But `recalibrate` sets the four cut points at the
20/40/60/80th percentiles of the file's own changes, so end-bin mass comes out
at 0.20 + 0.20 = 0.4000 for any continuous change distribution. Measured: sine
through zero 1.0000 -> 0.4004, iid gaussian -> 0.40, mixed-scale garbage ->
0.5714. The highest post-refit value found over seven deliberately hostile
inputs was 0.5714. The condition `end2 > 0.90` was dead code, so the branch was
unreachable by construction rather than merely unlucky.

WHAT THE REFUSAL TESTS NOW
--------------------------
Whether the refitted ladder is a usable measurement scale at all, which is the
question the refusal was always trying to ask:

  1. the four cut points must be STRICTLY increasing -- a tie means a bin of
     zero width and a ladder that has collapsed onto a point mass;
  2. the change distribution must hold at least MIN_DISTINCT distinct values --
     five bins cannot be placed on a distribution with four;
  3. post-refit end-bin mass must come below SAT_LIMIT.

Falsified, not asserted -- four inputs that make the refusal print, and two
that correctly do not (see WORKFLOW_VERIFY_2026-08-26.md for the full table).
"""
import numpy as np

WIN = 8
TICKS = [-0.002, -0.0005, 0.0005, 0.002]
SAT_LIMIT = 0.90
#: Distinct change values needed before a five-bin ladder can be placed. Four
#: bins need four cut points; asking a distribution with three values to
#: support them produces ties, and a tied ladder is not a scale.
MIN_DISTINCT = 10


def channel_values(vecs, c):
    """Every raw sample this channel contributed, across all windows."""
    out = [v[c * WIN:(c + 1) * WIN] for _, v in vecs]
    return np.concatenate(out) if out else np.array([])


def channel_changes(vecs, c):
    """Fractional sample-to-sample changes for ONE channel across all windows."""
    out = []
    for _, v in vecs:
        sl = v[c * WIN:(c + 1) * WIN]
        sl = sl[sl != 0]
        if len(sl) > 1:
            out.append(np.diff(sl) / (np.abs(sl[:-1]) + 1e-10))
    if not out:
        return np.array([])
    ch = np.concatenate(out)
    return ch[np.isfinite(ch)]


def n_distinct(ch, rtol=1e-9):
    """Distinct change values, counted at a RELATIVE tolerance.

    Exact `np.unique` is the wrong tool on floats. A channel that steps by
    exactly x1.5 every sample produces changes that are all 0.5 in arithmetic
    but differ in the last bits once they are computed at different magnitudes,
    so an exact count returned hundreds of "distinct" values for a distribution
    with one. Found while falsifying this gate on 2026-08-26.
    """
    if not len(ch):
        return 0
    v = np.sort(np.asarray(ch, float))
    scale = np.maximum(np.abs(v[:-1]), np.abs(v[1:]))
    gap = np.abs(np.diff(v)) > rtol * np.maximum(scale, 1e-300)
    return int(1 + np.count_nonzero(gap))


def occupancy_of(ch, ticks=None):
    t = TICKS if ticks is None else ticks
    if not len(ch):
        return np.zeros(5)
    return np.bincount(np.digitize(ch, t), minlength=5)[:5] / len(ch)


def ladder_from(ch, qs=(0.2, 0.4, 0.6, 0.8)):
    """Refit the four cut points to this channel's own change distribution."""
    return list(np.quantile(ch, qs)) if len(ch) else None


def gate_channel(vecs, c, name):
    """Health of one channel. Returns a dict; never a bare verdict."""
    ch = channel_changes(vecs, c)
    r = {"channel": name, "n_changes": int(len(ch)),
         "distinct": n_distinct(ch),
         "levels": 0, "end": float("nan"), "refit_end": None, "ticks": None,
         "status": "", "why": ""}
    if not len(ch):
        r["status"] = "NO DATA"
        r["why"] = "no usable sample-to-sample changes in this channel"
        return r
    occ = occupancy_of(ch)
    r["end"] = float(occ[0] + occ[4])

    # THE RESOLUTION TEST RUNS FIRST, BEFORE ANY EARLY RETURN. It used to sit
    # behind `if end <= SAT_LIMIT: return "readable"`, so the refusal could only
    # fire on a channel that ALSO happened to saturate the fixed ladder. A
    # binary 0/1 status column -- the canonical case the refusal text describes
    # -- came back "readable" with end-bin mass 0.0000, because `sl[sl != 0]` in
    # channel_changes deletes every zero sample and what is left is a constant.
    # That is analyze.py's own documented failure mode ("a flat channel returns
    # a clean, believable null that means nothing") reproduced inside the gate
    # written to prevent it. All four original falsification inputs happened to
    # saturate first, so none of them exercised this path. Found by DooDoo
    # 2026-08-26 and reproduced before fixing.
    #
    # Counted on the channel's own SAMPLE VALUES, exactly, with no tolerance:
    # a sensor logged at five levels has five levels whatever the float noise.
    n_levels = int(np.unique(channel_values(vecs, c)).size)
    r["levels"] = n_levels
    if n_levels < MIN_DISTINCT:
        r["status"] = "REFUSED"
        r["why"] = (f"it takes only {n_levels} distinct value(s) across the "
                    f"whole record. That is a channel logged at a resolution "
                    f"coarser than any movement worth reading — a status flag, "
                    f"a stepped setpoint, or a sensor quantised past the point "
                    f"where a five-bin ladder means anything")
        return r

    if r["end"] <= SAT_LIMIT:
        r["status"] = "readable"
        return r

    if r["distinct"] < MIN_DISTINCT:
        r["status"] = "REFUSED"
        r["why"] = (f"its sample-to-sample changes take only {r['distinct']} "
                    f"distinct value(s); a five-bin measurement ladder cannot be "
                    f"placed on that, so no scale exists on which this channel "
                    f"can be read")
        return r
    ticks = ladder_from(ch)
    if ticks is None or any(b <= a for a, b in zip(ticks, ticks[1:])):
        r["status"] = "REFUSED"
        r["ticks"] = ticks
        r["why"] = ("refitting the ladder to this channel's own changes "
                    "produces cut points that are not strictly increasing "
                    f"({['%.6g' % t for t in ticks] if ticks else None}) — the "
                    "scale collapses onto a point mass, which usually means "
                    "consecutive rows are not consecutive measurements of the "
                    "same quantity")
        return r
    occ2 = occupancy_of(ch, ticks)
    r["refit_end"] = float(occ2[0] + occ2[4])
    r["ticks"] = ticks
    if r["refit_end"] > SAT_LIMIT:
        r["status"] = "REFUSED"
        r["why"] = (f"saturated at {r['end']:.4f} on the standard scale and "
                    f"still {r['refit_end']:.4f} after refitting to its own "
                    f"change distribution")
        return r
    r["status"] = "refitted"
    return r


def gate_all(vecs, names, nch):
    return [gate_channel(vecs, c, names[c] if c < len(names) else f"slot{c+1}")
            for c in range(nch)]
