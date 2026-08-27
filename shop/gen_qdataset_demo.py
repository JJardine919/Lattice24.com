#!/usr/bin/env python3
"""
Regenerate a reduced slice of QDataSet G_1q_X_Z_N1 (Perrier, Youssry & Ferrie 2021,
MIT licence, DOI 10.5281/zenodo.5202814) using their published parameters:
Omega=12, T=1, M=1024 time steps, 5 Gaussian sigma_x pulses, noise profile N1
(1/f power spectrum + bump) driving sigma_z. Pure numpy reimplementation of
their simulation/utilites pipeline (their TF layer is acceleration only).
Output: one CSV time series suitable for the Lattice24 shop.
"""
import numpy as np

rng = np.random.default_rng(20260824)

# --- published parameters ---
DIM      = 2
OMEGA    = 12.0
T        = 1.0
M        = 1024
NUMPULSE = 5
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
H0   = 0.5 * OMEGA * SZ
HX   = 0.5 * SX
HNZ  = 0.5 * SZ
psi0 = np.array([[0.70710678], [0.70710678]], complex)  # |+> shows dephasing

# --- control: 5 gaussian pulses (random amps / centres / widths) ---
amps   = rng.uniform(0.5, 1.5, NUMPULSE)
mus    = np.sort(rng.uniform(0.15 * T, 0.85 * T, NUMPULSE))
sigmas = rng.uniform(0.02, 0.06, NUMPULSE)
t      = (np.arange(M) + 0.5) * (T / M)
u      = sum(a * np.exp(-((t - mu) ** 2) / (2 * sg ** 2))
             for a, mu, sg in zip(amps, mus, sigmas))

# --- noise N1: 1/f PSD with flat tail + bump, random-phase iFFT (their recipe) ---
f     = np.fft.rfftfreq(M, d=T / M)
S_Z   = (1 / (f + 1)) * (f <= 15) + (1 / 16) * (f > 15) \
        + np.exp(-((f - 30) ** 2) / 50) / 2
phase = np.sqrt(S_Z * M / (T / M)) * np.exp(2j * np.pi * rng.random(len(f)))
spec  = np.concatenate([phase, np.flip(phase.conj()[1:-1])])
beta  = np.real(np.fft.ifft(spec))

# --- evolution: piecewise-constant propagators, Pauli expectations ---
def expm_h(H):
    w, v = np.linalg.eigh(H)
    return (v * np.exp(-1j * w)) @ v.conj().T

ex = {p: [] for p in "xyz"}
for k in range(M):
    Hk = H0 + u[k] * HX + beta[k] * HNZ
    U  = expm_h(Hk * (T / M))
    psi = U @ psi0
    ex['x'].append(np.real(psi.conj().T @ SX @ psi)[0, 0])
    ex['y'].append(np.real(psi.conj().T @ SY @ psi)[0, 0])
    ex['z'].append(np.real(psi.conj().T @ SZ @ psi)[0, 0])

out = "t,pulse_u_t,noise_beta_z,sigma_x_expect,sigma_y_expect,sigma_z_expect\n"
out += "\n".join(f"{t[k]:.6f},{u[k]:.6f},{beta[k]:.6f},"
                 f"{ex['x'][k]:.6f},{ex['y'][k]:.6f},{ex['z'][k]:.6f}"
                 for k in range(M)) + "\n"
open("/tmp/opencode/qdataset_demo.csv", "w").write(out)
print("wrote /tmp/opencode/qdataset_demo.csv", len(out), "bytes")
print("sigma_z range:", min(ex['z']), max(ex['z']), "| beta sd:", round(float(np.std(beta)), 3))
