"""The shared control harness. Every adapter goes through this, or not at all.

Two paths, two questions:
  screen_window(w)          -- does the ORDER of these 24 numbers carry
                               information? Arms C0-C3 plus IAAFT and
                               permutation entropy.
  screen_populations(X, y)  -- does this labelled COMPARISON between groups
                               carry information? Label-shuffle, matched random
                               projections, and trivial engine-free columns,
                               every arm handed byte-identical preprocessed data.
"""
from .controls import (Verdict, ScreenResult, instrument_gate, screen_window,
                       PASS_Z, RP_Z, DEFAULT_SHUFFLES)
from .populations import (PopVerdict, PopResult, screen_populations,
                          dose_ladder, preprocess, collapse_features,
                          score_auc, score_maxz)
from .surrogates import (iaaft_surrogate, iaaft_pool, perm_entropy,
                         pe_is_degenerate, spectrum_fidelity)

__all__ = ["Verdict", "ScreenResult", "instrument_gate", "screen_window",
           "PASS_Z", "RP_Z", "DEFAULT_SHUFFLES",
           "PopVerdict", "PopResult", "screen_populations", "dose_ladder",
           "preprocess", "collapse_features", "score_auc", "score_maxz",
           "iaaft_surrogate", "iaaft_pool", "perm_entropy", "pe_is_degenerate",
           "spectrum_fidelity"]
