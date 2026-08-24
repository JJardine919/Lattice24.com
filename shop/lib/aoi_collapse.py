"""
AOI Collapse Core — Unified collapse for DooDoo (AOI v3.0)

All capabilities (speech, driving, personality) emerge from one
24D octonion/Leech decomposition. No separate agents.

Math: Zenodo DOI chain 18690444, 18722487, 18809406, 18809716.
"""

import numpy as np


# ============================================================
# Cayley multiplication table for octonions (64 entries)
# Basis: e0=1, e1..e7 imaginary
# Fano plane triples: (1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)
# Entry (i,j) = (sign, result_index) for e_i * e_j
# ============================================================

_CAYLEY_TABLE = [
    #        e0      e1      e2      e3      e4      e5      e6      e7
    [( 1,0),( 1,1),( 1,2),( 1,3),( 1,4),( 1,5),( 1,6),( 1,7)],  # e0
    [( 1,1),(-1,0),( 1,4),( 1,7),(-1,2),( 1,6),(-1,5),(-1,3)],  # e1
    [( 1,2),(-1,4),(-1,0),( 1,5),( 1,1),(-1,3),( 1,7),(-1,6)],  # e2
    [( 1,3),(-1,7),(-1,5),(-1,0),( 1,6),( 1,2),(-1,4),( 1,1)],  # e3
    [( 1,4),( 1,2),(-1,1),(-1,6),(-1,0),( 1,7),( 1,3),(-1,5)],  # e4
    [( 1,5),(-1,6),( 1,3),(-1,2),(-1,7),(-1,0),( 1,1),( 1,4)],  # e5
    [( 1,6),( 1,5),(-1,7),( 1,4),(-1,3),(-1,1),(-1,0),( 1,2)],  # e6
    [( 1,7),( 1,3),( 1,6),(-1,1),( 1,5),(-1,4),(-1,2),(-1,0)],  # e7
]

# Precompute 8x8x8 structure tensor for vectorized multiplication
# _MUL_TENSOR[k,i,j] = sign if e_i*e_j = sign*e_k, else 0
_MUL_TENSOR = np.zeros((8, 8, 8), dtype=np.float64)
for _i in range(8):
    for _j in range(8):
        _s, _k = _CAYLEY_TABLE[_i][_j]
        _MUL_TENSOR[_k, _i, _j] = _s


class Octonion:
    """8D octonion: e0 (real) + e1..e7 (imaginary)."""

    __slots__ = ('v',)

    def __init__(self, components):
        self.v = np.asarray(components, dtype=np.float64).ravel()
        if len(self.v) < 8:
            self.v = np.pad(self.v, (0, 8 - len(self.v)))
        elif len(self.v) > 8:
            self.v = self.v[:8]

    @staticmethod
    def random(rng=None):
        rng = rng or np.random.default_rng()
        return Octonion(rng.standard_normal(8))

    @staticmethod
    def zero():
        return Octonion(np.zeros(8))

    @property
    def real(self):
        return self.v[0]

    @property
    def vec(self):
        """Imaginary components e1..e7."""
        return self.v[1:]

    def norm(self):
        return np.linalg.norm(self.v)

    def conjugate(self):
        c = self.v.copy()
        c[1:] = -c[1:]
        return Octonion(c)

    def __add__(self, other):
        return Octonion(self.v + other.v)

    def __sub__(self, other):
        return Octonion(self.v - other.v)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return Octonion(self.v * other)
        # result[k] = sum_{i,j} T[k,i,j] * a[i] * b[j]
        return Octonion(np.einsum('kij,i,j->k', _MUL_TENSOR, self.v, other.v))

    def __rmul__(self, scalar):
        return Octonion(self.v * scalar)

    def __truediv__(self, scalar):
        return Octonion(self.v / float(scalar))

    def __neg__(self):
        return Octonion(-self.v)

    def dot(self, other):
        """Euclidean inner product on R^8."""
        return float(np.dot(self.v, other.v))

    def __repr__(self):
        return f"Oct({', '.join(f'{x:.4f}' for x in self.v)})"


# ============================================================
# Jordan-Shadow Decomposition (DOI 18690444)
# ============================================================

def octonion_shadow_decompose(A: Octonion, B: Octonion) -> dict:
    """
    Jordan-Shadow decomposition (DOI 18690444).

    Jordan     = (AB + BA)/2  — symmetric intent (rational)
    Commutator = (AB - BA)/2  — anti-symmetric directional (motor/steer)
    Associator = J * C        — interaction chaos (personality)

    J + C = AB exactly (lossless). The associator captures how the
    symmetric and anti-symmetric parts interact through the non-commutative
    octonion product — non-zero in general, bounded, no third element needed.

    Verifies on imaginary parts (.vec):
        <J.vec, C.vec> ≈ 0           (orthogonality)
        ||AB.vec||² ≈ ||J.vec||² + ||C.vec||²  (Pythagorean)
    """
    AB = A * B
    BA = B * A

    jordan = (AB + BA) / 2
    commutator = (AB - BA) / 2

    # Associator: octonion product of Jordan and Commutator
    # Measures non-linear interaction between symmetric/anti-symmetric parts.
    # Non-zero whenever A,B don't commute (generic case).
    associator = jordan * commutator

    # Verify orthogonality on imaginary parts
    # C.real = 0 always (AB.real = BA.real for octonions), so <J.vec, C.vec> = 0
    jc_dot = np.dot(jordan.vec, commutator.vec)
    assert abs(jc_dot) < 1e-8, f"Orthogonality violated: <J.vec,C.vec> = {jc_dot}"

    # Verify Pythagorean on imaginary parts (exact since C.real = 0)
    ab_n2 = np.linalg.norm(AB.vec) ** 2
    jc_n2 = np.linalg.norm(jordan.vec) ** 2 + np.linalg.norm(commutator.vec) ** 2
    assert abs(ab_n2 - jc_n2) < 1e-8, \
        f"Pythagorean violated: {ab_n2:.6f} != {jc_n2:.6f}"

    return {
        'jordan': jordan,
        'commutator': commutator,
        'associator': associator,
        'product': AB,
    }


# ============================================================
# Entropy Transponders (DOIs 18809406, 18809716)
# ============================================================

def entropy_transponders(state_vec: np.ndarray,
                         num_transponders: int = 32) -> np.ndarray:
    """
    Entropy gating: 9 foundational + 17 adaptive + 6 evolutionary.

    Low-entropy dimensions pass through (stable signal).
    High-entropy dimensions get attenuated (chaotic noise).
    Evolutionary rotation breaks symmetry via entropy-weighted phase.
    """
    state = np.asarray(state_vec, dtype=np.float64).ravel()
    n = len(state)

    # --- 9 Foundational: per-dimension Shannon entropy ---
    shifted = state - np.max(state)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    probs = np.clip(probs, 1e-12, None)
    entropy_per_dim = -probs * np.log2(probs)
    global_entropy = float(np.sum(entropy_per_dim))

    # --- 17 Adaptive: threshold gating ---
    median_ent = np.median(entropy_per_dim)
    gate = np.where(
        entropy_per_dim <= median_ent,
        1.0,
        np.exp(-(entropy_per_dim - median_ent))
    )
    gated = state * gate

    # --- 6 Evolutionary: entropy-weighted Givens rotation ---
    if n >= 2:
        phase = global_entropy * np.pi / n
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        evolved = gated.copy()
        for k in range(0, n - 1, 2):
            a, b = evolved[k], evolved[k + 1]
            evolved[k] = cos_p * a - sin_p * b
            evolved[k + 1] = sin_p * a + cos_p * b
        return evolved

    return gated


# ============================================================
# TTS Prompt Builder
# ============================================================

# Calibrated from randn(24) inputs across scales 0.1–5.0:
#   scale=0.3 -> chaos ~0.17 (calm)
#   scale=1.0 -> chaos ~26   (mid)
#   scale=2.0 -> chaos ~564  (wild)
# Divide by 5.0 so unit-norm inputs land at ~5 (mid-band)
_CHAOS_SCALE = 5.0

_CHAOS_BANDS = [
    (2.0, "calm and thoughtful"),
    (5.0, "curious and playful"),
    (8.0, "wildly enthusiastic and erratic"),
    (float('inf'), "totally unhinged and bursting with energy"),
]


def _build_tts_prompt(chaos_raw: float, jordan_mean: float,
                      intent_mag: float, control_norm: float) -> str:
    """Build a TTS-ready prompt string from collapse outputs."""
    normalized = min(chaos_raw / _CHAOS_SCALE, 10.0)

    # Chaos description from bands
    chaos_desc = _CHAOS_BANDS[-1][1]
    for threshold, desc in _CHAOS_BANDS:
        if normalized < threshold:
            chaos_desc = desc
            break

    # Intent hint from Jordan mean
    if abs(jordan_mean) > 0.5:
        sign = "positive" if jordan_mean > 0 else "negative"
        intent_hint = f"focused on {sign} intent ({intent_mag:.2f} clarity)"
    else:
        intent_hint = "exploring freely"

    # Control flavor from commutator norm
    control_flavor = ""
    if control_norm > 5.0:
        control_flavor = ", with quick directional bursts"

    return (
        f"Speak as DooDoo, an excited chaotic artificial organism "
        f"who's {chaos_desc} about {intent_hint}{control_flavor}."
    )


# ============================================================
# Unified AOI Collapse
# ============================================================

def aoi_collapse(high_dim_state: np.ndarray) -> dict:
    """
    One collapse timestep. Input 24D state -> outputs emerge.

    Pipeline:
        entropy gate -> octonion projection (3x8D) -> decompose -> extract

    Returns:
        personality_embedding: 8D associator vector (non-associative chaos)
        chaos_level:           scalar norm of associator
        control_vec:           3D directional (steer, throttle, brake)
        intent_magnitude:      scalar clarity of rational intent
        text_prompt_base:      string for TTS/personality conditioning
    """
    state = np.asarray(high_dim_state, dtype=np.float64).ravel()

    # Pad/truncate to 24D (Leech-inspired)
    if len(state) < 24:
        state = np.pad(state, (0, 24 - len(state)))
    else:
        state = state[:24]

    # 1. Entropy gate
    gated = entropy_transponders(state)

    # 2. Project to octonion pair: A(0:8), B(8:16)
    #    Remaining dims 16:24 modulate B (context scaling)
    A = Octonion(gated[:8])
    B = Octonion(gated[8:16])
    context = gated[16:24]
    ctx_scale = 1.0 + np.linalg.norm(context) * 0.1
    B = B * ctx_scale

    # 3. Jordan-Shadow decompose (A, B only — no third element)
    decomp = octonion_shadow_decompose(A, B)

    jordan = decomp['jordan']
    commutator = decomp['commutator']
    associator = decomp['associator']

    # 4. Extract outputs
    personality_embedding = associator.v
    chaos_raw = associator.norm()
    control_vec = commutator.vec[:3]  # e1,e2,e3 -> steer, throttle, brake
    control_norm = float(np.linalg.norm(control_vec))
    intent_mag = float(np.mean(np.abs(jordan.vec)))
    jordan_mean = float(np.mean(jordan.vec))

    # 5. Build TTS-ready text_prompt_base
    text_prompt_base = _build_tts_prompt(chaos_raw, jordan_mean, intent_mag,
                                         control_norm)

    return {
        'personality_embedding': personality_embedding,
        'chaos_level': chaos_raw,
        'normalized_chaos': min(chaos_raw / _CHAOS_SCALE, 10.0),
        'control_vec': control_vec,
        'control_norm': control_norm,
        'intent_magnitude': intent_mag,
        'text_prompt_base': text_prompt_base,
        'decomposition': decomp,
    }


# ============================================================
# Standalone verification
# ============================================================

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    passed = 0
    total = 7

    print("=" * 60)
    print("AOI Collapse Core — Verification")
    print("=" * 60)

    # Test 1: Cayley table correctness
    print("\n[1] Cayley table")
    e = [Octonion(np.eye(8)[i]) for i in range(8)]
    ok = True
    # e1*e2 = e4
    ok &= np.allclose((e[1] * e[2]).v, e[4].v)
    # e2*e1 = -e4 (anti-commutative)
    ok &= np.allclose((e[2] * e[1]).v, -e[4].v)
    # e_i^2 = -1 for i>0
    for i in range(1, 8):
        ok &= np.allclose((e[i] * e[i]).v, -e[0].v)
    # Spot checks from Fano triples
    ok &= np.allclose((e[3] * e[4]).v, e[6].v)   # (3,4,6)
    ok &= np.allclose((e[5] * e[6]).v, e[1].v)   # (5,6,1)
    ok &= np.allclose((e[7] * e[1]).v, e[3].v)   # (7,1,3)
    if ok:
        print("    PASS")
        passed += 1
    else:
        print("    FAIL")

    # Test 2: Norm multiplicativity |AB| = |A||B|
    print("\n[2] Norm multiplicativity")
    ok = True
    for _ in range(10):
        A = Octonion.random(rng)
        B = Octonion.random(rng)
        lhs = (A * B).norm()
        rhs = A.norm() * B.norm()
        if abs(lhs - rhs) > 1e-8:
            ok = False
            print(f"    FAIL: |AB|={lhs:.8f} != |A||B|={rhs:.8f}")
            break
    if ok:
        print("    PASS")
        passed += 1

    # Test 3: Jordan-Shadow decomposition
    print("\n[3] Decomposition (orthogonality + Pythagorean on .vec)")
    ok = True
    for trial in range(10):
        A = Octonion.random(rng)
        B = Octonion.random(rng)
        try:
            decomp = octonion_shadow_decompose(A, B)
        except AssertionError as ex:
            print(f"    FAIL trial {trial}: {ex}")
            ok = False
            break
        j, c = decomp['jordan'], decomp['commutator']
        # Verify reconstruction: J + C = AB
        recon = j + c
        if not np.allclose(recon.v, decomp['product'].v, atol=1e-10):
            print(f"    FAIL: J + C != AB")
            ok = False
            break
        # Print verification details for first trial
        if trial == 0:
            print(f"    <J.vec, C.vec> = {np.dot(j.vec, c.vec):.2e}")
            ab_n2 = np.linalg.norm(decomp['product'].vec) ** 2
            jc_n2 = np.linalg.norm(j.vec) ** 2 + np.linalg.norm(c.vec) ** 2
            print(f"    ||AB.vec||² = {ab_n2:.6f}")
            print(f"    ||J.vec||² + ||C.vec||² = {jc_n2:.6f}")
    if ok:
        print("    PASS")
        passed += 1

    # Test 4: Non-zero associator (J*C captures non-commutativity)
    print("\n[4] Associator (J*C) non-zero check")
    nonzero = 0
    for _ in range(20):
        A = Octonion.random(rng)
        B = Octonion.random(rng)
        decomp = octonion_shadow_decompose(A, B)
        anorm = decomp['associator'].norm()
        if anorm > 1e-10:
            nonzero += 1
    print(f"    Non-zero associators: {nonzero}/20")
    if nonzero > 0:
        print("    PASS")
        passed += 1
    else:
        print("    FAIL — associator always zero")

    # Test 5: Entropy transponders
    print("\n[5] Entropy transponders")
    state = rng.standard_normal(24)
    gated = entropy_transponders(state)
    ratio = np.linalg.norm(gated) / np.linalg.norm(state)
    print(f"    Input norm:  {np.linalg.norm(state):.4f}")
    print(f"    Gated norm:  {np.linalg.norm(gated):.4f}")
    print(f"    Attenuation: {ratio:.2%}")
    if 0 < ratio <= 1.5:  # gating shouldn't amplify wildly
        print("    PASS")
        passed += 1
    else:
        print("    FAIL — gating ratio out of range")

    # Test 6: Full AOI collapse
    print("\n[6] Full AOI collapse")
    ok = True
    for trial in range(3):
        state = rng.standard_normal(24) * (1 + trial)
        try:
            result = aoi_collapse(state)
        except Exception as ex:
            print(f"    FAIL trial {trial}: {ex}")
            ok = False
            break
        print(f"    Trial {trial}:")
        print(f"      Chaos:   {result['chaos_level']:.4f} (normalized: {result['normalized_chaos']:.2f})")
        print(f"      Control: [{', '.join(f'{x:.3f}' for x in result['control_vec'])}] norm={result['control_norm']:.3f}")
        print(f"      Intent:  {result['intent_magnitude']:.4f}")
    if ok:
        print("    PASS")
        passed += 1

    # Test 7: TTS prompt variation across chaos bands
    print("\n[7] TTS prompt modulation")
    ok = True
    seen_descs = set()
    for trial in range(50):
        state = rng.standard_normal(24)
        scale = rng.uniform(0.1, 5.0)
        result = aoi_collapse(state * scale)
        prompt = result['text_prompt_base']
        # Extract chaos description
        for _, desc in _CHAOS_BANDS:
            if desc in prompt:
                seen_descs.add(desc)
                break
    print(f"    Chaos bands hit: {len(seen_descs)}/4")
    for desc in sorted(seen_descs):
        print(f"      - {desc}")
    # Show a few example prompts
    print("\n    Example prompts:")
    for scale_label, scale_val in [("quiet", 0.3), ("normal", 1.0), ("intense", 4.0)]:
        state = rng.standard_normal(24) * scale_val
        result = aoi_collapse(state)
        print(f"      [{scale_label}] {result['text_prompt_base']}")
    if len(seen_descs) >= 2:
        print("    PASS")
        passed += 1
    else:
        print("    FAIL — not enough prompt variation")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed")
    if passed == total:
        print("ALL VERIFICATIONS PASSED")
    print("=" * 60)
