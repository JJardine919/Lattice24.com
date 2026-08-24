#!/usr/bin/env python3
"""brainrow, with two defects fixed. Import this instead of copying drift_oilgas.py.

Both defects were found on 2026-08-05 while testing the full stack on public
Climate TRACE CO2e data, and both are in the plumbing, not the algebra.

FIX 1 — the transponder -> collapse handoff violated the one-scale rule.
  THE_ENGINE.md says "One scale per vector. Mixing counts in the thousands with
  scores under 1 produced a false -2.5 sigma twice." The original brainrow packs
  16 Shannon entropies (bounded, ~0.4) into the same 24D state as
  signal_strength = mean(|features_128|), which is UNBOUNDED and tracks the input
  scale. Measured on 1,655 power plants: one slot held 69% of the normalised 24D
  state on average and 100% of it for some plants -- i.e. aoi_collapse was being
  handed one number and 23 near-zeros.

  Two candidate fixes were tested, and ONLY ONE WORKS. Do not confuse them.

  (a) Per-row magnitude fix -- apply the engine's own convention,
      `min(signal_strength/100.0, 1.0)` (oilgas_brain.py:349), inside the
      handoff too. Needs no dataset and cuts mean dominance 0.66 -> 0.24.
      **It does not recover the signal**: oil p = 1.0000 -> 0.97, i.e. still
      no better than random projections. Available as onescale=True.

  (b) Per-slot VARIANCE fix -- standardise each of the 24 slots using means and
      SDs fitted across a sample of rows (SlotScaler below), so every slot
      carries comparable variation between samples rather than merely comparable
      size. **This is the one that works**: oil p = 1.0000 -> 0.0000,
      waste stays 0.0000, gas 1.0000 -> 0.10.

  The lesson is sharper than THE_ENGINE.md's "one scale per vector": the collapse
  needs each slot to vary comparably ACROSS SAMPLES. A slot that is large but
  nearly constant still contributes nothing, and a slot that is small but
  informative is still swamped. Magnitude parity is not enough.

  Verified fold-safe: fitting SlotScaler on the training fold only and applying
  it to the held-out fold reproduces the transductive numbers (oil +0.0731 vs
  +0.0727, waste +0.0426 vs +0.0428), so this is deployable, not leakage.

FIX 2 — the quantum layer was unseeded.
  AerSimulator.run() was called without seed_simulator, so identical input gave
  expanded_probability anywhere in 0.430..0.520 across repeat calls. The
  TransponderEngine is deterministic; only this was not. Sampling noise is now
  removed by seeding, so a rerun reproduces exactly.

Effect of FIX 1, measured, replicated twice on independent quantum draws
(1,655 US power plants, fuel-type classification, vs 100 matched random
projections of the same inputs):
    oil   p = 1.0000  ->  p = 0.0000
    gas   p = 1.0000  ->  p = 0.09 / 0.12
    waste p = 0.0000  ->  p = 0.0000   (already positive, unchanged)
    coal  null both ways
"""
import importlib.util as _u
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from aoi_collapse import aoi_collapse  # noqa: E402

OILGAS = str(__import__("pathlib").Path(__file__).resolve().parent / "oilgas_brain.py")
SEED = 20260805
SIGNAL_SCALE = 100.0     # the engine's own convention, oilgas_brain.py:349

_spec = _u.spec_from_file_location("ogb", OILGAS)
_ogb = _u.module_from_spec(_spec)
_spec.loader.exec_module(_ogb)

# ---- FIX 2: make the Aer sampler reproducible -------------------------------
try:
    from qiskit_aer import AerSimulator
except ImportError:                                   # older layout
    from qiskit.providers.aer import AerSimulator

if not getattr(AerSimulator, "_voodoo_seeded", False):
    _orig_run = AerSimulator.run

    def _seeded_run(self, *a, **k):
        k.setdefault("seed_simulator", SEED)
        return _orig_run(self, *a, **k)

    AerSimulator.run = _seeded_run
    AerSimulator._voodoo_seeded = True

engine = _ogb.TransponderEngine()
quantum = _ogb.QuantumProcessor(engine)
assert abs(engine.compute_super_logarithm() - 34.031437) < 1e-5, \
    "wrong engine -- super logarithm must be 34.031437"


def state24(sr, onescale=True):
    """Build the 24D collapse input from a transponder result.

    onescale=True applies FIX 1. Pass False only to reproduce the old behaviour.
    """
    sig = sr["signal_strength"]
    if onescale:
        sig = min(sig / SIGNAL_SCALE, 1.0)
    return np.array(sr["sensor_entropies"] +
                    [sr["mean_entropy"], sr["entropy_spread"], sig,
                     sr["max_entropy"], sr["min_entropy"]] + [0, 0, 0])[:24]


class SlotScaler:
    """Fix (b): put all 24 collapse slots on comparable variance across samples.

    Fit on training rows only, then transform anything. This is what actually
    recovers the collapse layer's contribution; the per-row magnitude fix does not.

        sc = SlotScaler().fit([state24(sr, onescale=False) for sr in train_srs])
        r  = aoi_collapse(sc.collapse_input(state24(sr, onescale=False)))
    """

    def fit(self, states):
        S = np.asarray(states, dtype=float)
        self.mu_ = S.mean(0)
        self.sd_ = S.std(0)
        self.keep_ = self.sd_ > 1e-12
        return self

    def transform(self, st):
        z = np.zeros(24)
        z[self.keep_] = (np.asarray(st, float)[self.keep_] - self.mu_[self.keep_]) \
            / self.sd_[self.keep_]
        return z

    def collapse_input(self, st):
        z = self.transform(st)
        n = np.linalg.norm(z)
        return z / n if n > 0 else z


def brainrow(v128, shots=256, onescale=True, return_dominance=False, scaler=None):
    """Transponders -> quantum -> collapse. Returns the 27-feature row."""
    sr = engine.process_sample(np.asarray(v128, dtype=float))
    q = quantum.run(sr, shots=shots)
    if scaler is not None:
        st = state24(sr, onescale=False)
        un = scaler.collapse_input(st)
    else:
        st = state24(sr, onescale)
        n = np.linalg.norm(st)
        un = st / n if n > 0 else st
    r = aoi_collapse(un)
    row = (list(sr["sensor_entropies"]) +
           [sr["mean_entropy"], sr["entropy_spread"], sr["signal_strength"],
            sr["max_entropy"], sr["min_entropy"],
            q["anomaly_probability"], q["compressed_probability"],
            q["expanded_probability"],
            r["chaos_level"], r["intent_magnitude"], r["control_norm"]])
    if return_dominance:
        return row, float(np.max(np.abs(un)))
    return row


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    v = rng.normal(size=24)
    v = v * (10.0 / np.linalg.norm(v))
    v128 = np.resize(v, 128)

    a, b = brainrow(v128), brainrow(v128)
    print("FIX 2 determinism (identical input, two calls):",
          "PASS" if np.allclose(a, b) else "FAIL")

    _, dom_old = brainrow(v128, onescale=False, return_dominance=True)
    _, dom_new = brainrow(v128, onescale=True, return_dominance=True)
    print(f"FIX 1 dominance of largest slot: {dom_old:.3f} -> {dom_new:.3f}")
