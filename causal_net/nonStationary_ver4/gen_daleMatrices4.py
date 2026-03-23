#!/usr/bin/env python3
"""
 ./gen_daleMatrices4.py --num_neurons 100 --num_excite 80 --placement_H_L_delta 1 2 2 --placement_min_dist 0.01 --dataName test1

Primary purpose: generate the ground-truth dictionary (A_true, B_true) for use
by gen_nonStationarySpikes3.py.  The stationary spike generation performed here
is for evaluation only (firing-rate sanity check per B-vector).

Topology (writeup): grid placement on [0,L]×[0,H], distance-kernel affinities,
fixed random out-degree per neuron, Dale's law (column sign = presynaptic type),
negative self-loops, E–I weight scaling, spectral radius normalization.

CLI placement: --placement_H_L_delta (H, L, placement_ker_delta), --placement_min_dist (grid spacing); out-degree
fraction range: --edge_prob lo hi (maps to k_min, k_max).

Pipeline:
1. Build rectangular grid (spacing placement_min_dist); sample N nodes without replacement;
   sort positions by x. Assign N_exc excitatory / rest inhibitory at random.
2. Affinity V_ij = D_ij^{-δ} with placement_ker_delta from --placement_H_L_delta; sample k_j targets per column j without
   replacement with probabilities proportional to V_ij.
3. Signed topology E (diagonal −1); raw weights W from Uniform(1−v,1+v) with
   inhibitory scale −r U; rescale so rho(A_true) = R.

Neuron index order follows placement (sorted by x, then y). Type labels are in
tau (0=exc, 1=inh); B vectors and rate summaries use tau, not index blocks.

Output files saved to <basePath>/truthDale/:
  <dataName>.simTruth.npz   — A_true, B_true, E_true, node_positions, node_types, … + metadata
  <dataName>.spikes.npz     — stationary spikes + rate statistics

Output shapes (N = num_neurons, T = num_steps):
  node_positions (N, 2)   float  — (x,y) sorted by x then y
  node_types    (N,)      int    — 0 excitatory, 1 inhibitory
  node_signs    (N,)      int8   — +1 / −1 Dale signs (presynaptic)
  node_out_degrees (N,)   int    — out-degree (topology)
  E_true        (N, N)    int8   — signed topology {−1,0,+1}
  A_true        (N, N)    float  — weighted connectivity, rho(A_true)=R
  B_true        (N,)      float  — bias vector
  node_distance_matrix (N, N) float — pairwise Euclidean distances; diagonal 0
  offdiag_kernel       (mem_lag_steps,) float — κ_ℓ off-diagonal memory weights (model B only)
  spikes        (T, N)    uint8  — spike counts

Spike GLM (--spike_model): A = lag-1 (baseline Y[0] Poisson); B = lag-M
off-diagonal kernel (--mem_lag_steps required for B, --time_kernel_a_q_tau); τ_mem in seconds;
tau_mem_steps = round(τ_mem/dt) (int); default τ_mem=(4/3)*mem_lag_steps*dt if τ_mem<=0; for each lag ℓ use damped oscillator only if ℓ≤(3/4)τ_mem_steps, else κ_ℓ=0; first mem_lag_steps bins zero for B.
"""

import numpy as np
import time
import hashlib
import os
import sys

import argparse
from pprint import pprint

from toolbox.Util_NumpyIO import write_data_npz
from UtilDalePoisson4 import estimate_rates

###### Matrix generation (spatial + Dale, see writeup) ##################


def _build_placement_grid(L, H, d_min):
    """Rectangular grid nodes on [0,L]×[0,H] with spacing d_min."""
    m_max = int(np.floor(L / d_min))
    n_max = int(np.floor(H / d_min))
    gx = (np.arange(0, m_max + 1, dtype=float) * d_min).reshape(-1, 1)
    gy = (np.arange(0, n_max + 1, dtype=float) * d_min).reshape(1, -1)
    xs = np.broadcast_to(gx, (gx.size, gy.size)).ravel()
    ys = np.broadcast_to(gy, (gx.size, gy.size)).ravel()
    G = np.column_stack([xs, ys])
    return G, (m_max + 1) * (n_max + 1)


def _sample_targets_without_replacement(j, V_col, k_j, rng):
    """Draw k_j distinct row indices i≠j with P(i) ∝ V_ij."""
    n = V_col.shape[0]
    p = np.asarray(V_col, dtype=float).copy()
    p[j] = 0.0
    s = p.sum()
    if s <= 0:
        raise RuntimeError("zero affinity sum for presynaptic neuron %d" % j)
    p /= s
    return rng.choice(n, size=k_j, replace=False, p=p)


def generate_spatial_dale_network(
    n_units,
    n_excite,
    L,
    H,
    d_min,
    k_min,
    k_max,
    placement_ker_delta,
    weight_var,
    R,
    rng=None,
    verb=1,
):
    """
    Returns A_true, E_true, P, tau, sigma, k_out, rho0, D_save in placement order
    (x-sorted). D_save has pairwise Euclidean distances and zeros on the diagonal.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_inhib = n_units - n_excite
    assert n_excite > 0 and n_inhib > 0
    assert k_min >= 1 and k_max <= n_units - 1 and k_min <= k_max
    assert placement_ker_delta in (1, 2)
    assert 0 < weight_var < 1
    assert 0 < R < 1

    r_balance = n_excite / float(n_inhib)

    G, n_grid = _build_placement_grid(L, H, d_min)
    if n_grid < n_units:
        raise ValueError(
            "grid has only %d nodes; need N <= N_G (reduce d_min or increase L/H)"
            % n_grid
        )

    idx = rng.choice(n_grid, size=n_units, replace=False)
    P = G[idx].astype(np.float64)
    order_x = np.lexsort((P[:, 1], P[:, 0]))
    P = P[order_x]

    exc_mask = np.zeros(n_units, dtype=bool)
    exc_mask[rng.choice(n_units, size=n_excite, replace=False)] = True
    tau = np.where(exc_mask, 0, 1).astype(np.int32)
    sigma = np.where(exc_mask, 1, -1).astype(np.int8)

    # D_ij = distance row i to column j (postsynaptic i, presynaptic j)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, np.inf)
    with np.errstate(divide="ignore"):
        V = np.power(D, -float(placement_ker_delta))
    np.fill_diagonal(V, 0.0)

    E = np.zeros((n_units, n_units), dtype=np.int8)
    k_out = np.zeros(n_units, dtype=np.int32)

    for j in range(n_units):
        kj = int(rng.integers(k_min, k_max + 1))
        k_out[j] = kj
        targets = _sample_targets_without_replacement(j, V[:, j], kj, rng)
        E[targets, j] = sigma[j]
        E[j, j] = -1

    W = np.zeros((n_units, n_units), dtype=np.float64)
    for j in range(n_units):
        for i in range(n_units):
            eij = E[i, j]
            if eij == 0:
                continue
            u = float(rng.uniform(1.0 - weight_var, 1.0 + weight_var))
            if i == j:
                W[i, j] = -u
            elif eij == 1:
                W[i, j] = u
            else:
                W[i, j] = -r_balance * u

    ev = np.linalg.eigvals(W)
    rho0 = float(np.max(np.abs(ev)))
    if verb > 0:
        print("initW current_rho :", rho0)
    if rho0 <= 0:
        raise RuntimeError("spectral radius of W is zero (degenerate)")
    A_true = W * (R / rho0)

    D_save = D.copy()
    np.fill_diagonal(D_save, 0.0)

    return A_true, E, P, tau, sigma, k_out, rho0, D_save


def spectral_radius_scaling(A, factors):
    n = A.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)

    print(f"{'factor':>8s}  {'ρ(all)':>10s}  {'ρ(off-diag)':>12s}")
    print("-" * 34)

    for c in factors:
        rho_all = np.max(np.abs(np.linalg.eigvals(c * A)))

        A_off = A.copy()
        A_off[off_diag_mask] *= c
        rho_off = np.max(np.abs(np.linalg.eigvals(A_off)))

        print(f"{c:8.3f}  {rho_all:10.4f}  {rho_off:12.4f}")


def summarize_pairwise_distances(D, verb=1):
    """
    Mean and median Euclidean distance over unique unordered pairs (i < j).
    D: (N, N) symmetric nonnegative; diagonal entries are ignored.
    Returns dict with mean, median, n_nodes, n_pairs.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix")
    n = D.shape[0]
    if n < 2:
        out = {"mean": float("nan"), "median": float("nan"), "n_nodes": n, "n_pairs": 0}
        if verb > 0:
            print("Pairwise distance: N<2 — no pairs (mean/median undefined).")
        return out
    iu = np.triu_indices(n, k=1)
    d = D[iu]
    m = float(np.mean(d))
    med = float(np.median(d))
    out = {"mean": m, "median": med, "n_nodes": n, "n_pairs": int(d.size)}
    if verb > 0:
        print(
            "Pairwise distance (unique pairs): mean=%.6g  median=%.6g  (N=%d, pairs=%d)"
            % (m, med, n, out["n_pairs"])
        )
    return out


#################### Simulation ##################


def set_flat_selfSpiking(Nn, idleRate, spect_radius, tau):
    """Bias vector (N,); per-neuron E/I from tau (0=exc, 1=inh)."""
    tau = np.asarray(tau).reshape(-1)
    if tau.shape[0] != Nn:
        raise ValueError("tau length %d != Nn %d" % (tau.shape[0], Nn))
    if tau.dtype.kind not in "iu":
        raise ValueError("tau must be integer dtype")
    exc = tau.astype(np.int64) == 0
    inh = tau.astype(np.int64) == 1
    if not np.any(exc) or not np.any(inh):
        raise ValueError("tau must label at least one excitatory and one inhibitory neuron")
    sizeScale = np.sqrt(float(Nn) / 100)
    R = float(spect_radius)
    idle_eff = np.array(idleRate, dtype=float)
    Ri_scaled = idle_eff * R
    Bi = np.log(Ri_scaled)
    B = np.random.uniform(Bi[0], Bi[1], size=(Nn,))
    #B[exc] += 1.0 - R * 1.5 # - sizeScale
    B[exc] += -2.0 
    return B


def gen_stationary_lag1_poisson(num_steps, dt, A, B_intercept, tau, eta_clip, verb=0):
    """
    Generates a multivariate Poisson VAR(1) process:
        Y_t ~ Poisson(exp(A @ Y_{t-1} + B_intercept))
    tau: (N,) int, 0=excitatory, 1=inhibitory (same order as A columns / Y).
    """
    if A is None:
        raise ValueError("Connectivity matrix A cannot be None.")
    d = A.shape[0]
    tau = np.asarray(tau).reshape(-1)
    if tau.shape[0] != d:
        raise ValueError("tau length %d != A.shape[0] %d" % (tau.shape[0], d))
    if tau.dtype.kind not in "iu":
        raise ValueError("tau must be integer dtype")
    tau_i = tau.astype(np.int64)
    if np.any((tau_i != 0) & (tau_i != 1)):
        raise ValueError("tau entries must be 0 or 1")
    exc_idx = np.where(tau_i == 0)[0]
    inh_idx = np.where(tau_i == 1)[0]
    n_exc = len(exc_idx)
    if verb > 0:
        print(f"\n=== Generating Poisson  Process ===")
        print(f"Simulation parameters: steps={num_steps}, dt={dt:.3f}, neurons={d}, excit={n_exc}")
        print(f"Matrix A stats: min={np.min(A):.3f}, max={np.max(A):.3f}, mean={np.mean(A):.3f}")
        print(f"Bias B stats: min={np.min(B_intercept):.3f}, max={np.max(B_intercept):.3f}, mean={np.mean(B_intercept):.3f}")

    Y = np.zeros((num_steps, d), dtype=int)
    Y[0] = np.random.poisson(np.exp(B_intercept) * dt)  # initial state
    if verb > 0:
        print(
            f"Initial state: total spikes={np.sum(Y[0])}, excit spikes={np.sum(Y[0, exc_idx])}, inhib spikes={np.sum(Y[0, inh_idx])}"
        )

    kk = min(5, n_exc, len(inh_idx))
    if verb > 0 and kk > 0:
        eix = exc_idx[:kk]
        iix = inh_idx[:kk]
        print(
            "t=0 Y[t] sum=%d, Excit(sample idx %s):%s, Inhib(sample idx %s):%s"
            % (np.sum(Y[0]), eix, Y[0, eix], iix, Y[0, iix])
        )

    if verb > 0:
        print("Starting main simulation loop...")
    for t in range(1, num_steps):
        eta = A @ Y[t - 1] + B_intercept
        lambda_t = np.exp(np.clip(eta, max=eta_clip))  # avoid overflow
        Y[t] = np.random.poisson(lambda_t * dt)

        if verb > 0 and t < 5 and kk > 0:
            eix = exc_idx[:kk]
            iix = inh_idx[:kk]
            print(
                "t=%d Y[t] sum=%d, Excit(sample):%s, Inhib(sample):%s"
                % (t, np.sum(Y[t]), Y[t, eix], Y[t, iix])
            )

        if verb > 0 and t % (num_steps // 4) == 0:
            print(f"  Progress: {t}/{num_steps} steps ({t/num_steps*100:.1f}%) -  total spikes in this step: {np.sum(Y[t])}")

    if verb > 0:
        print(f"Simulation complete. Final state: total spikes={np.sum(Y[-1])}")
        print(f"Spike data stats: min={np.min(Y)}, max={np.max(Y)}, mean={np.mean(Y):.2f}, total spikes={np.sum(Y)}")

    return Y


def _offdiag_memory_kernel(mem_lag_steps, tau_mem_steps, Q_mem, amp_mem):
    """
    Off-diagonal damped-oscillator: κ_ℓ = amp_mem * exp(-γℓ) cos(ωℓ) for ℓ ≤ (3/4)*τ_mem_steps; else κ_ℓ = 0.
    tau_mem_steps: period in bins (int).
    Returns kappa shape (mem_lag_steps,).
    """
    mem_lag_steps = int(mem_lag_steps)
    if mem_lag_steps < 1:
        raise ValueError("mem_lag_steps must be >= 1")
    tau_mem_steps = int(tau_mem_steps)
    if tau_mem_steps < 1:
        raise ValueError("tau_mem_steps must be >= 1")
    Q_mem = float(Q_mem)
    amp_mem = float(amp_mem)
    if Q_mem <= 0:
        raise ValueError("Q_mem must be positive")
    if amp_mem <= 0:
        raise ValueError("amp_mem must be positive")
    tau_f = float(tau_mem_steps)
    omega = 2.0 * np.pi / tau_f
    gamma = np.pi / (Q_mem * tau_f)
    # ℓ active iff ℓ·4 ≤ 3·τ_mem_steps (same as ℓ ≤ (3/4)·τ_mem_steps). Max such ℓ:
    ell_max_active = (3 * tau_mem_steps) // 4
    if ell_max_active < 1:
        print(
            "  minimal mem_lag_steps for non-zero κ: impossible — "
            "need τ_mem_steps ≥ 2 so that lag ℓ=1 can satisfy ℓ·4 ≤ 3·τ_mem_steps"
        )
    else:
        print(
            "  minimal mem_lag_steps for any non-zero κ: 1; "
            "minimal to include full oscillator support (all ℓ with κ≠0): %d"
            % ell_max_active
        )
    ell_i = np.arange(1, mem_lag_steps + 1, dtype=np.int64)
    active = ell_i * 4 <= 3 * tau_mem_steps
    ell_f = ell_i.astype(np.float64)
    kappa = np.zeros(mem_lag_steps, dtype=np.float64)
    kappa[active] = amp_mem * np.exp(-gamma * ell_f[active]) * np.cos(omega * ell_f[active])
    m = kappa.shape[0]
    n_act = int(np.sum(active))
    print(f"  κ (len={m}, oscillator lags ℓ≤(3/4)τ_mem_steps: {n_act}): {kappa}")
    sum_abs = float(np.sum(np.abs(kappa)))
    print(f"  sum_k |κ_k| = {sum_abs:.12g}")
    if sum_abs <= 0:
        print("  κ: all lags clipped — zero off-diagonal kernel")
    return kappa



def gen_stationary_lagM_modelB_poisson(
    num_steps,
    dt,
    A,
    B_intercept,
    tau,
    mem_lag_steps,
    tau_mem_steps,
    Q_mem,
    amp_mem,
    eta_clip,
    verb=0,
    h_off=None,
):
    """
    Model B: H_t = diag(A) Y_{t-1} + A_off (sum_l h_l Y_{t-l}), then Poisson as Model A.
    First mem_lag_steps bins are zero (pre-history).
    tau_mem_steps: oscillation period in bins (int).     If h_off is None, κ from (mem_lag_steps, tau_mem_steps, Q_mem, amp_mem)
    with κ_ℓ = 0 for lag ℓ > (3/4)*tau_mem_steps.
    """
    if A is None:
        raise ValueError("Connectivity matrix A cannot be None.")
    d = A.shape[0]
    mem_lag_steps = int(mem_lag_steps)
    tau = np.asarray(tau).reshape(-1)
    if tau.shape[0] != d:
        raise ValueError("tau length %d != A.shape[0] %d" % (tau.shape[0], d))
    if tau.dtype.kind not in "iu":
        raise ValueError("tau must be integer dtype")
    tau_i = tau.astype(np.int64)
    if np.any((tau_i != 0) & (tau_i != 1)):
        raise ValueError("tau entries must be 0 or 1")
    exc_idx = np.where(tau_i == 0)[0]
    inh_idx = np.where(tau_i == 1)[0]
    n_exc = len(exc_idx)

    if h_off is None:
        h_off = _offdiag_memory_kernel(mem_lag_steps, tau_mem_steps, Q_mem, amp_mem)
    else:
        h_off = np.asarray(h_off, dtype=np.float64).reshape(-1)
        if h_off.shape[0] != mem_lag_steps:
            raise ValueError("h_off length %d != mem_lag_steps %d" % (h_off.shape[0], mem_lag_steps))
    A_off = A.copy()
    np.fill_diagonal(A_off, 0.0)
    a_diag = np.diag(A).astype(np.float64, copy=False)

    if verb > 0:
        print(f"\n=== Generating Model B (lag-M) Poisson ===")
        print(
            f"steps={num_steps}, dt={dt:.3f}, neurons={d}, excit={n_exc}, mem_lag_steps={mem_lag_steps}, tau_mem_steps={tau_mem_steps}, Q_mem={Q_mem}, amp_mem={amp_mem}"
        )
        print(f"Matrix A stats: min={np.min(A):.3f}, max={np.max(A):.3f}, mean={np.mean(A):.3f}")
        print(f"Bias B stats: min={np.min(B_intercept):.3f}, max={np.max(B_intercept):.3f}, mean={np.mean(B_intercept):.3f}")

    Y = np.zeros((num_steps, d), dtype=int)
    if mem_lag_steps > 0:
        Y[:mem_lag_steps] = 0

    kk = min(5, n_exc, len(inh_idx))
    if verb > 0:
        print("Initial Y[0:%d] set to zero (pre-history)." % min(mem_lag_steps, num_steps))

    if verb > 0:
        print("Starting main simulation loop...")
    for t in range(1, num_steps):
        S = np.zeros(d, dtype=np.float64)
        for ell in range(1, mem_lag_steps + 1):
            tt = t - ell
            if tt >= 0:
                S += h_off[ell - 1] * Y[tt]
        eta = a_diag * Y[t - 1] + A_off @ S + B_intercept
        lambda_t = np.exp(np.clip(eta, max=eta_clip))
        Y[t] = np.random.poisson(lambda_t * dt)

        if verb > 0 and t < 5 and kk > 0:
            eix = exc_idx[:kk]
            iix = inh_idx[:kk]
            print(
                "t=%d Y[t] sum=%d, Excit(sample):%s, Inhib(sample):%s"
                % (t, np.sum(Y[t]), Y[t, eix], Y[t, iix])
            )

        if verb > 0 and t % (num_steps // 4) == 0:
            print(f"  Progress: {t}/{num_steps} steps ({t/num_steps*100:.1f}%) -  total spikes in this step: {np.sum(Y[t])}")

    if verb > 0:
        print(f"Simulation complete. Final state: total spikes={np.sum(Y[-1])}")
        print(f"Spike data stats: min={np.min(Y)}, max={np.max(Y)}, mean={np.mean(Y):.2f}, total spikes={np.sum(Y)}")

    return Y


#########################
#  MAIN
#########################


def main():
    print("=" * 60)
    print("DALE POISSON SIMULATION (Model A or B)")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Simulate a recurrent neural network with Dale's principle.")
    parser.add_argument(
        "--spike_model",
        type=str,
        default="A",
        choices=["A", "B"],
        help="Spike GLM: A = lag-1 stationary; B = lag-M off-diagonal damped-oscillator kernel.",
    )
    parser.add_argument(
        "--mem_lag_steps",
        type=int,
        default=None,
        help="Model B (required): number of off-diagonal history lags M (time steps). Ignored for model A.",
    )
    parser.add_argument(
        "--time_kernel_a_q_tau",
        type=float,
        nargs=3,
        default=[1.0, 3.0, 0.0],
        metavar=("amp_mem", "Q_mem", "tau_mem"),
        help="Model B: [amp_mem, Q_mem, tau_mem] — kernel amplitude, quality factor, period (s). If tau_mem<=0, use (4/3)*mem_lag_steps*step_size.",
    )
    parser.add_argument("--num_neurons", type=int, default=50, help="Total number of neurons in the network.")
    parser.add_argument("--num_excite", type=int, default=None, help="Number of excitatory neurons.")
    parser.add_argument(
        "--placement_H_L_delta",
        type=float,
        nargs=3,
        default=[1.0, 2.0, 2.0],
        metavar=("placement_H", "placement_L", "placement_ker_delta"),
        help="Placement: [0,H] height, [0,L] width, distance-kernel exponent δ in V_ij ∝ D_ij^{-δ} (1 or 2).",
    )
    parser.add_argument(
        "--placement_min_dist",
        type=float,
        default=0.01,
        help="Grid spacing d_min; minimum inter-neuron distance.",
    )
    parser.add_argument(    "--edge_prob",  type=float,   nargs=2,   default=[0.05, 0.2],
                            help="Out-degree range as fractions of N: k_min=max(1,floor(lo*N)), k_max=min(N-1,floor(hi*N)).",
                        )
    parser.add_argument("--weight_var", type=float, default=0.2, help="Fractional weight variation v for Uniform(1-v,1+v).")
    parser.add_argument("--num_steps", type=int, default=10_001, help="Number of time steps for simulation.")
    parser.add_argument("--step_size", type=float, default=0.01, help="Integration time step size (dt) in seconds.")
    parser.add_argument("--spectral_radius", type=float, default=0.9, help="Target spectral radius value for the connectivity matrix.")
    parser.add_argument("--idleRate", type=float, nargs=2, default=[15, 30.0], help="Range of idle firing rates [min, max] in Hz.")
    parser.add_argument("-v", "--verb", type=int, default=1, help="Verbosity level (0=quiet, 1=normal).")
    parser.add_argument("--dataName", type=str, default=None, help="Base name for output files (default: daleN<num_neurons>_<hash>).")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="Output directory for all files.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")

    np.set_printoptions(precision=3, suppress=True)

    args = parser.parse_args()
    placement_H, placement_L, placement_ker_delta_raw = (
        float(args.placement_H_L_delta[0]),
        float(args.placement_H_L_delta[1]),
        float(args.placement_H_L_delta[2]),
    )
    if not (placement_H > 0 and placement_L > 0):
        raise ValueError("placement_H_L_delta requires positive H and L")
    placement_ker_delta = int(round(placement_ker_delta_raw))
    if placement_ker_delta not in (1, 2):
        raise ValueError("placement_H_L_delta third value (placement_ker_delta) must be 1 or 2")
    placement_min_dist = float(args.placement_min_dist)
    if placement_min_dist <= 0:
        raise ValueError("placement_min_dist must be positive")
    if args.spike_model == "B" and args.mem_lag_steps is None:
        raise ValueError("--mem_lag_steps is required for spike_model B")
    amp_mem, Q_mem, tau_mem_arg = args.time_kernel_a_q_tau
    if tau_mem_arg <= 0:
        if args.mem_lag_steps is None:
            tau_mem = None
        else:
            tau_mem = (4.0 / 3.0) * float(args.mem_lag_steps) * float(args.step_size)
    else:
        tau_mem = float(tau_mem_arg)
    args.varTwindow = 5  # (sec)
    args.poisson_eta_clip = 5  # ~ [1e-3Hz , 1e+3Hz]
    if args.dataName is None:
        args.dataName = "daleN%d_" % args.num_neurons + hashlib.md5(os.urandom(32)).hexdigest()[:6]

    outPath = os.path.join(args.basePath, "truthDale")
    if args.num_excite is None:
        args.num_excite = int(0.8 * args.num_neurons)

    Nn = args.num_neurons
    prob_lo, prob_hi = float(args.edge_prob[0]), float(args.edge_prob[1])
    if not (0 < prob_lo < prob_hi <= 1.0):
        raise ValueError("edge_prob must satisfy 0 < lo < hi <= 1")
    k_out_min = max(1, int(np.floor(prob_lo * Nn)))
    k_out_max = int(np.floor(prob_hi * Nn))
    k_out_max = max(k_out_min, k_out_max)
    k_out_max = min(k_out_max, Nn - 1)

    if args.seed is not None:
        np.random.seed(args.seed)

    rng = np.random.default_rng(args.seed)

    print("gen dale matrices args:", vars(args), "\n")

    assert Nn >= 10
    assert args.num_excite >= 5
    assert args.num_excite < Nn
    assert os.path.exists(args.basePath)
    assert os.path.exists(outPath)
    assert args.num_steps >= 1000
    assert args.step_size > 0.001
    assert args.idleRate[0] >= 0.5
    assert args.idleRate[1] > args.idleRate[0]

    tau_mem_steps = None
    if args.spike_model == "B":
        if args.mem_lag_steps is None or args.mem_lag_steps < 1:
            raise ValueError("mem_lag_steps must be >= 1 for spike_model B")
        if tau_mem <= 0 or Q_mem <= 0 or amp_mem <= 0:
            raise ValueError("tau_mem, Q_mem, and amp_mem must be positive for spike_model B")
        if args.mem_lag_steps > args.num_steps:
            raise ValueError("mem_lag_steps cannot exceed num_steps")
        tau_mem_steps = max(1, int(round(float(tau_mem) / float(args.step_size))))

    verb_r = args.verb
    print(f"\n{'='*60}")
    print(f"  Spectral radius: R={args.spectral_radius:.3f}")
    print(f"{'='*60}")

    A_dale, E_true, P_pos, tau, sigma, k_out, rho0, D_mat = generate_spatial_dale_network(
        n_units=Nn,
        n_excite=args.num_excite,
        L=placement_L,
        H=placement_H,
        d_min=placement_min_dist,
        k_min=k_out_min,
        k_max=k_out_max,
        placement_ker_delta=placement_ker_delta,
        weight_var=args.weight_var,
        R=args.spectral_radius,
        rng=rng,
        verb=verb_r,
    )

    if verb_r > 0:
        summarize_pairwise_distances(D_mat, verb=verb_r)

    if 0:
        factors = [0.2, 0.4, 0.6, 0.8, 0.9]
        spectral_radius_scaling(A_dale, factors)
        sys.exit(1)

    if verb_r > 0:
        total_connections = A_dale.size
        zero_connections = np.sum(np.abs(A_dale) < 1e-10)
        non_zero_connections = total_connections - zero_connections
        sparsity = zero_connections / total_connections
        print(f"Matrix sparsity: {sparsity*100:.1f}% ({zero_connections}/{total_connections} connections are zero)")
        print(f"Non-zero connections: {non_zero_connections} ({(1-sparsity)*100:.1f}%)")

    dale_conf = {
        "num_neurons": args.num_neurons,
        "num_excite": args.num_excite,
        "spectral_radius": args.spectral_radius,
        "placement_L": placement_L,
        "placement_H": placement_H,
        "placement_min_dist": placement_min_dist,
        "edge_prob": [prob_lo, prob_hi],
        "k_out_min": k_out_min,
        "k_out_max": k_out_max,
        "placement_ker_delta": placement_ker_delta,
        "weight_var": args.weight_var,
        "rho0_unscaled": rho0,
        "idleRate": args.idleRate,
        "spike_model": args.spike_model,
    }
    if args.spike_model == "B":
        dale_conf["mem_lag_steps"] = args.mem_lag_steps
        dale_conf["tau_mem"] = tau_mem
        dale_conf["tau_mem_steps"] = tau_mem_steps
        dale_conf["Q_mem"] = Q_mem
        dale_conf["amp_mem"] = amp_mem
    if args.seed is not None:
        dale_conf["seed"] = args.seed

    evol_conf = {
        "num_steps": args.num_steps,
        "step_size": args.step_size,
        "evol_time": args.num_steps * args.step_size,
        "poisson_eta_clip": args.poisson_eta_clip,
        "spike_model": args.spike_model,
    }
    if args.spike_model == "B":
        evol_conf["mem_lag_steps"] = args.mem_lag_steps
        evol_conf["tau_mem"] = tau_mem
        evol_conf["tau_mem_steps"] = tau_mem_steps
        evol_conf["Q_mem"] = Q_mem
        evol_conf["amp_mem"] = amp_mem

    print("Dale configuration:")
    pprint(dale_conf)

    B_true = set_flat_selfSpiking(Nn, args.idleRate, args.spectral_radius, tau)

    max_samples = 100_000

    print(f"\n{'='*60}")
    print("  Poisson evaluation spike generation (model %s)" % args.spike_model)
    print(f"{'='*60}")

    offdiag_kernel = None
    if args.spike_model == "B":
        offdiag_kernel = _offdiag_memory_kernel(
            args.mem_lag_steps, tau_mem_steps, Q_mem, amp_mem
        )
        
    start_time = time.time()
    if args.spike_model == "A":
        Y = gen_stationary_lag1_poisson(
            num_steps=args.num_steps,
            dt=args.step_size,
            A=A_dale,
            B_intercept=B_true,
            tau=tau,
            eta_clip=args.poisson_eta_clip,
            verb=verb_r,
        )
    else:
        Y = gen_stationary_lagM_modelB_poisson(
            num_steps=args.num_steps,
            dt=args.step_size,
            A=A_dale,
            B_intercept=B_true,
            tau=tau,
            mem_lag_steps=args.mem_lag_steps,
            tau_mem_steps=tau_mem_steps,
            Q_mem=Q_mem,
            amp_mem=amp_mem,
            eta_clip=args.poisson_eta_clip,
            verb=verb_r,
            h_off=offdiag_kernel,
        )
    sim_time = time.time() - start_time
    print("Spike generation completed in %.1f seconds" % sim_time)

    stats_dict, rates_dict, _ = estimate_rates(
        Y,
        args.step_size,
        tau,
        max_samples,
        args.spectral_radius,
        varTwindow=args.varTwindow,
        mxNn=5,
        verb=0,
    )
    stats_dict["var_time_window_sec"] = float(args.varTwindow)
    stats_dict["max_samples"] = int(max_samples)

    Y_u8 = np.clip(Y, 0, 255).astype(np.uint8)

    trueD = {
        "A_true": A_dale,
        "B_true": B_true,
        "E_true": E_true,
        "node_positions": P_pos,
        "node_is_inhibitory": tau,
        "node_distance_matrix": D_mat,
    }
    if args.spike_model == "B":
        trueD["offdiag_kernel"] = offdiag_kernel

    trueMD = {
        "dale_conf": dale_conf,
        "evol_conf": evol_conf,
        "short_name": args.dataName,
        "provenance": {"state_model_file": args.dataName},
    }

    spikeD = {
        "spikes": Y_u8,
        "single_rates": rates_dict["single_rates"],
        "sigle_rates_var": rates_dict["sigle_rates_var"],
        "single_fano_fact": rates_dict["single_fano_fact"],
    }
    spikeMD = {
        "short_name": args.dataName,
        "time_step_sec": args.step_size,
        "data_type": "simDaleStates",
        "poisson_eta_clip": args.poisson_eta_clip,
        "spike_model": args.spike_model,
        "placement_L": float(placement_L),
        "placement_H": float(placement_H),
        "placement_min_dist": float(placement_min_dist),
    }
    if args.spike_model == "B":
        spikeMD["mem_lag_steps"] = args.mem_lag_steps
        spikeMD["tau_mem"] = tau_mem
        spikeMD["tau_mem_steps"] = tau_mem_steps
        spikeMD["Q_mem"] = Q_mem
        spikeMD["amp_mem"] = amp_mem

    outFt = os.path.join(outPath, args.dataName + ".simTruth.npz")
    write_data_npz(trueD, outFt, metaD=trueMD)
    if args.verb > 1:
        pprint(trueMD)
    outFs = os.path.join(outPath, args.dataName + ".spikes.npz")
    write_data_npz(spikeD, outFs, metaD=spikeMD)
    if args.verb > 1:
        pprint(spikeMD)

    print("\nSimulation completed successfully!")
    print(f"\nRate Summary {args.dataName}  N={args.num_neurons}  exc={args.num_excite}, R={args.spectral_radius:.3f}  ")
    s = stats_dict
    print(
        f"  rates:  all={s['avg_spike_rate_all']:14.1f}  exc={s['avg_spike_rate_excit']:14.1f}  inh={s['avg_spike_rate_inhib']:14.1f}"
    )
    print("\nNext step commands:")
    print("     basePath=" + args.basePath)
    print("  ./view_daleMatrix4.py  --basePath $basePath   --dataName %s  -p a e b  c d f -X  " % args.dataName)
    print("  ./view_spikesTrain4.py  --basePath $basePath   --dataName %s  --time_range_sec 1 8 -p b --time_rebin2 2   -X " % args.dataName)
    print("  ./view_spikesTrain4.py  --basePath $basePath   --dataName %s  --time_range_sec 0 20 -p b   -X " % args.dataName)
    print("  ./gen_nonStationarySpikes3.py  --basePath $basePath   --inputStates %s     --true_dwell_sec 1.0 " % args.dataName)
    print("  ./fit_lassoPoisson.py  --basePath $basePath  --inpPath ${basePath}/truthDale --dataName   %s   --num_epochs  50  " % args.dataName)


if __name__ == "__main__":
    main()
