#!/usr/bin/env python3
"""
Generate synthetic spike trains for the Real-Time Adaptive Bernoulli
State-Space Network Model (BSSM) with short-term synaptic depression (STD)
and a lag-M synaptic kernel.

Output files:

  <basePath>/truthDale/<dataName>.simTruth.npz
  <basePath>/truthDale/<dataName>.spikes.npz

The one architectural exception requested by the user is preserved:
the connectivity matrix is generated from 2D spatial placement exactly in the
style of the current Dale generator, but the diagonal is forced to zero so the
network has no self-loops.
"""

import argparse
import os
import sys
import time
from pprint import pprint

import numpy as np


from toolbox.Util_NumpyIO import write_data_npz
from UtilDalePoisson5 import estimate_rates


if sys.version_info < (3, 0):
    sys.stderr.write("ERROR: gen5_BSSM_STD_spikes.py requires Python 3.0 or newer.\n")
    sys.exit(1)


def _build_placement_grid(length_x, height_y, d_min):
    """Rectangular grid nodes on [0,L] x [0,H] with spacing d_min."""
    m_max = int(np.floor(length_x / d_min))
    n_max = int(np.floor(height_y / d_min))
    gx = (np.arange(0, m_max + 1, dtype=float) * d_min).reshape(-1, 1)
    gy = (np.arange(0, n_max + 1, dtype=float) * d_min).reshape(1, -1)
    xs = np.broadcast_to(gx, (gx.size, gy.size)).ravel()
    ys = np.broadcast_to(gy, (gx.size, gy.size)).ravel()
    grid = np.column_stack([xs, ys])
    return grid, (m_max + 1) * (n_max + 1)


def _sample_targets_without_replacement(j, affinity_column, k_j, rng):
    """Draw k_j distinct postsynaptic targets i != j with P(i) proportional to affinity."""
    n_units = affinity_column.shape[0]
    p = np.asarray(affinity_column, dtype=float).copy()
    p[j] = 0.0
    norm = p.sum()
    if norm <= 0:
        raise RuntimeError("zero affinity sum for presynaptic neuron %d" % j)
    p /= norm
    return rng.choice(n_units, size=k_j, replace=False, p=p)


def generate_spatial_dale_network(
    n_units,
    n_excite,
    length_x,
    height_y,
    d_min,
    k_min,
    k_max,
    placement_ker_delta,
    weight_var,
    spectral_radius_target,
    rng=None,
    verb=1,
):
    """
    Generate the spatial/Dale recurrent matrix used by the old generator,
    but with a zero diagonal (no self-loops).

    Returns
    -------
    W_true : (N, N)
        Spectrally scaled recurrent weight matrix in postsynaptic-row,
        presynaptic-column convention.
    sign_true : (N, N)
        Signed binary topology: +1 excitatory edge, -1 inhibitory edge, 0 absent.
    positions : (N, 2)
        Neuron positions sorted by x then y.
    tau : (N,)
        0 excitatory, 1 inhibitory.
    rho0 : float
        Spectral radius before rescaling.
    distance_matrix : (N, N)
        Pairwise Euclidean distances with zero diagonal.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_inhib = n_units - n_excite
    assert n_excite > 0 and n_inhib > 0
    assert k_min >= 1 and k_max <= n_units - 1 and k_min <= k_max
    assert float(placement_ker_delta) > 0
    assert 0 < weight_var < 1
    assert 0 < spectral_radius_target < 1

    balance_scale = n_excite / float(n_inhib)

    grid, n_grid = _build_placement_grid(length_x, height_y, d_min)
    if n_grid < n_units:
        raise ValueError(
            "grid has only %d nodes; need N <= N_G (reduce d_min or increase L/H)"
            % n_grid
        )

    sample_idx = rng.choice(n_grid, size=n_units, replace=False)
    positions = grid[sample_idx].astype(np.float64)
    order_x = np.lexsort((positions[:, 1], positions[:, 0]))
    positions = positions[order_x]

    exc_mask = np.zeros(n_units, dtype=bool)
    exc_mask[rng.choice(n_units, size=n_excite, replace=False)] = True
    tau = np.where(exc_mask, 0, 1).astype(np.int32)
    sigma = np.where(exc_mask, 1, -1).astype(np.int8)

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(distance_matrix, np.inf)
    with np.errstate(divide="ignore"):
        affinity = np.power(distance_matrix, -float(placement_ker_delta))
    np.fill_diagonal(affinity, 0.0)

    sign_true = np.zeros((n_units, n_units), dtype=np.int8)
    for j in range(n_units):
        kj = int(rng.integers(k_min, k_max + 1))
        targets = _sample_targets_without_replacement(j, affinity[:, j], kj, rng)
        sign_true[targets, j] = sigma[j]

    raw_W = np.zeros((n_units, n_units), dtype=np.float64)
    for j in range(n_units):
        edge_mask = sign_true[:, j] != 0
        if not np.any(edge_mask):
            continue
        weights = rng.uniform(1.0 - weight_var, 1.0 + weight_var, size=int(np.sum(edge_mask)))
        if sigma[j] < 0:
            weights *= -balance_scale
        raw_W[edge_mask, j] = weights

    np.fill_diagonal(raw_W, 0.0)

    eigvals = np.linalg.eigvals(raw_W)
    rho0 = float(np.max(np.abs(eigvals)))
    if verb > 0:
        print("initW current_rho :", rho0)
    if rho0 <= 0:
        raise RuntimeError("spectral radius of raw_W is zero (degenerate)")

    W_true = raw_W * (spectral_radius_target / rho0)
    np.fill_diagonal(W_true, 0.0)

    distance_save = distance_matrix.copy()
    np.fill_diagonal(distance_save, 0.0)
    return W_true, sign_true, positions, tau, rho0, distance_save


def summarize_pairwise_distances(distance_matrix, verb=1):
    """Mean and median Euclidean distance over unique unordered pairs (i < j)."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    n_units = distance_matrix.shape[0]
    if n_units < 2:
        raise ValueError("at least 2 neurons are required for pairwise distances")
    iu = np.triu_indices(n_units, k=1)
    distances = distance_matrix[iu]
    out = {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "n_nodes": int(n_units),
        "n_pairs": int(distances.size),
    }
    if verb > 0:
        print(
            "Pairwise distance (unique pairs): mean=%.6g  median=%.6g  (N=%d, pairs=%d)"
            % (out["mean"], out["median"], out["n_nodes"], out["n_pairs"])
        )
    return out


def _sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def _logit(prob):
    prob = np.asarray(prob, dtype=np.float64)
    return np.log(prob) - np.log1p(-prob)


def set_flat_selfSpiking(n_units, idle_rate_hz, dt, tau, rng):
    """
    Bernoulli baseline logits derived from target idle firing rates in Hz.

    The returned B_true is a log-odds vector. If recurrent input were zero,
    neuron i would spike with probability p_i ~ rate_i * dt in each bin.
    """
    tau = np.asarray(tau).reshape(-1)
    if tau.shape[0] != n_units:
        raise ValueError("tau length %d != n_units %d" % (tau.shape[0], n_units))
    if tau.dtype.kind not in "iu":
        raise ValueError("tau must be integer dtype")
    exc = tau.astype(np.int64) == 0
    inh = tau.astype(np.int64) == 1
    if not np.any(exc) or not np.any(inh):
        raise ValueError("tau must label at least one excitatory and one inhibitory neuron")

    idle_rate_hz = np.asarray(idle_rate_hz, dtype=float)
    if idle_rate_hz.shape != (2,):
        raise ValueError("idleRate must be a length-2 range [min_hz, max_hz]")
    if idle_rate_hz[0] <= 0 or idle_rate_hz[1] <= idle_rate_hz[0]:
        raise ValueError("idleRate must satisfy 0 < min_hz < max_hz")

    per_neuron_rate_hz = rng.uniform(idle_rate_hz[0], idle_rate_hz[1], size=n_units)
    p_bin = np.clip(per_neuron_rate_hz * dt, 1e-6, 1.0 - 1e-6)
    return _logit(p_bin)


def build_exponential_kernel(dt, synaptic_tau, mem_lag_steps, verb=0):
    """
    Exponential lag-M kernel:
        kappa_l = (1 - alpha) alpha^(l-1), alpha = exp(-dt / tau_s)

    This finite kernel is implemented with the equivalent tail-corrected
    recursion
        h_{t+1} = alpha h_t + (1 - alpha) u_t
                  - (1 - alpha) alpha^M u_{t-M}.
    """
    synaptic_tau = float(synaptic_tau)
    if synaptic_tau <= 0:
        raise ValueError("synaptic_tau must be positive")
    mem_lag_steps = int(mem_lag_steps)
    if mem_lag_steps < 1:
        raise ValueError("mem_lag_steps must be >= 1")

    alpha = float(np.exp(-dt / synaptic_tau))
    ell = np.arange(mem_lag_steps, dtype=np.float64)
    kernel = (1.0 - alpha) * np.power(alpha, ell)
    if verb > 0:
        print(
            "  exponential kernel: len=%d, span=%.6g sec, tau_s=%.6g sec, alpha=%.6g, sum=%.12g"
            % (mem_lag_steps, mem_lag_steps * float(dt), synaptic_tau, alpha, float(np.sum(kernel)))
        )
    return kernel, alpha


def gen_bssm_std_bernoulli(
    num_steps,
    dt,
    W_true,
    B_true,
    tau,
    std_u,
    std_tau_rec,
    synaptic_alpha,
    mem_lag_steps,
    logit_clip,
    rng,
    verb=0,
):
    """
    Simulate the BSSM-STD Bernoulli model.

    h_t is the filtered presynaptic drive available at the start of bin t.
    x_t is the available resource fraction at the start of bin t.
    """
    if W_true is None:
        raise ValueError("W_true cannot be None")
    n_units = W_true.shape[0]
    if W_true.shape != (n_units, n_units):
        raise ValueError("W_true must be square")
    tau = np.asarray(tau).reshape(-1)
    if tau.shape[0] != n_units:
        raise ValueError("tau length %d != W_true.shape[0] %d" % (tau.shape[0], n_units))
    if tau.dtype.kind not in "iu":
        raise ValueError("tau must be integer dtype")
    tau_i = tau.astype(np.int64)
    if np.any((tau_i != 0) & (tau_i != 1)):
        raise ValueError("tau entries must be 0 or 1")
    if B_true.shape != (n_units,):
        raise ValueError("B_true shape %s does not match (%d,)" % (B_true.shape, n_units))
    if not (0 < std_u <= 1):
        raise ValueError("std_u must satisfy 0 < std_u <= 1")
    if std_tau_rec <= 0:
        raise ValueError("std_tau_rec must be positive")
    if not (0 <= synaptic_alpha < 1):
        raise ValueError("synaptic_alpha must satisfy 0 <= alpha < 1")
    mem_lag_steps = int(mem_lag_steps)
    if mem_lag_steps < 1:
        raise ValueError("mem_lag_steps must be >= 1")

    x = np.ones(n_units, dtype=np.float64)
    h = np.zeros(n_units, dtype=np.float64)
    u_history = np.zeros((mem_lag_steps, n_units), dtype=np.float64)
    recovery_decay = float(np.exp(-dt / std_tau_rec))
    synaptic_gain = 1.0 - synaptic_alpha
    tail_gain = synaptic_gain * float(np.power(synaptic_alpha, mem_lag_steps))

    spikes = np.zeros((num_steps, n_units), dtype=np.uint8)
    x_true = np.zeros((num_steps, n_units), dtype=np.float32)
    u_true = np.zeros((num_steps, n_units), dtype=np.float32)
    h_true = np.zeros((num_steps, n_units), dtype=np.float32)
    p_true = np.zeros((num_steps, n_units), dtype=np.float32)
    n_exc = int(np.sum(tau_i == 0))
    progress_stride = max(1, num_steps // 4)

    if verb > 0:
        print("\n=== Generating BSSM-STD Bernoulli spikes ===")
        print(
            "steps=%d, dt=%.6g, neurons=%d, excit=%d, std_u=%.6g, tau_rec=%.6g, alpha=%.6g, gain=%.6g, M=%d"
            % (
                num_steps,
                dt,
                n_units,
                n_exc,
                std_u,
                std_tau_rec,
                synaptic_alpha,
                synaptic_gain,
                mem_lag_steps,
            )
        )
        print(
            "W stats: min=%.3f, max=%.3f, mean=%.3f"
            % (float(np.min(W_true)), float(np.max(W_true)), float(np.mean(W_true)))
        )
        print(
            "b stats: min=%.3f, max=%.3f, mean=%.3f"
            % (float(np.min(B_true)), float(np.max(B_true)), float(np.mean(B_true)))
        )

    t_start = time.time()
    for t in range(num_steps):
        old_u = u_history[t % mem_lag_steps].copy()
        x_true[t] = x
        h_true[t] = h

        logits_t = np.clip(B_true + W_true @ h, -logit_clip, logit_clip)
        p_t = _sigmoid(logits_t)
        s_t = (rng.random(n_units) < p_t).astype(np.uint8)
        u_t = std_u * x * s_t

        spikes[t] = s_t
        p_true[t] = p_t
        u_true[t] = u_t

        if verb > 0 and t < 5:
            print(
                "t=%d spikes=%d  mean_p=%.4f  mean_x=%.4f  mean_h=%.4f"
                % (t, int(np.sum(s_t)), float(np.mean(p_t)), float(np.mean(x)), float(np.mean(h)))
            )

        x_depleted = x * (1.0 - std_u * s_t)
        x = 1.0 - (1.0 - x_depleted) * recovery_decay
        x = np.clip(x, 0.0, 1.0)
        h = synaptic_alpha * h + synaptic_gain * u_t - tail_gain * old_u
        h = np.maximum(h, 0.0)
        u_history[t % mem_lag_steps] = u_t

        if verb > 0 and t > 0 and t % progress_stride == 0:
            elapsed = time.time() - t_start
            print(
                "  Progress: %d/%d steps (%.1f%%) - total spikes this bin: %d  elaT=%.1fs"
                % (t, num_steps, 100.0 * t / num_steps, int(np.sum(s_t)), elapsed)
            )

    if verb > 0:
        print("Simulation complete. Final total spikes=%d" % int(np.sum(spikes[-1])))
        print(
            "Spike data stats: min=%d, max=%d, mean=%.4f, total spikes=%d"
            % (
                int(np.min(spikes)),
                int(np.max(spikes)),
                float(np.mean(spikes)),
                int(np.sum(spikes)),
            )
        )

    out = {
        "spikes": spikes,
        "x_true": x_true,
        "u_true": u_true,
        "h_true": h_true,
        "p_true": p_true,
        "x_final": x.astype(np.float32),
        "h_final": h.astype(np.float32),
    }
    return out


def main():
    print("=" * 60)
    print("BSSM-STD BERNOULLI SIMULATION")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="Simulate the BSSM-STD Bernoulli network with spatial/Dale connectivity."
    )
    p = parser.add_argument
    p("--synaptic_tau", type=float, default=0.005, help="Synaptic decay constant tau_s in seconds.")
    p("--std_recovery_tau", type=float, default=0.300, help="STD recovery constant tau_rec in seconds.")
    p("--std_u", type=float, default=0.4, help="STD utilization U in (0, 1].")
    p("--kernel_len_steps", type=int, default=25,
      help="Lag-M synaptic-kernel length in bins.")
    p("--num_neurons", type=int, default=50, help="Total number of neurons in the network.")
    p("--num_excite", type=int, required=True, help="Number of excitatory neurons.")
    p("--placement_H_L_delta", type=float, nargs=3, default=[1.0, 2.0, 2.0],
      metavar=("placement_H", "placement_L", "placement_ker_delta"),
      help="Placement: [0,H] height, [0,L] width, and distance-kernel exponent delta > 0.")
    p("--placement_min_dist", type=float, default=0.01, help="Grid spacing d_min; minimum inter-neuron distance.")
    p("--edge_prob", type=float, nargs=2, default=[0.05, 0.2],
      help="Out-degree range as fractions of N: k_min=max(1,floor(lo*N)), k_max=min(N-1,floor(hi*N)).")
    p("--init_weight_var", type=float, default=0.4, help="Fractional weight variation v for Uniform(1-v,1+v).")
    p("--num_steps", type=int, default=5_001, help="Number of time steps for simulation.")
    p("--step_size", type=float, default=0.001, help="Time bin width dt in seconds.")
    p("--spectral_radius", type=float, default=0.90, help="Target spectral radius for the off-diagonal recurrent matrix.")
    p("--idleRate", type=float, nargs=2, default=[30.0, 50.0], help="Range of baseline firing rates [min, max] in Hz.")
    p("--logit_clip", type=float, default=20.0, help="Clip logits to [-logit_clip, +logit_clip].")
    p("-v", "--verb", type=int, default=1, help="Verbosity level (0=quiet, 1=normal).")
    p("--dataName", type=str, required=True, help="Base name for output files.")
    p("--basePath", type=str, default="/pscratch/sd/b/balewski/2026_causalNet_tmp/",
      help="Output directory root; files are written under <basePath>/truthDale/.")

    np.set_printoptions(precision=3, suppress=True)
    args = parser.parse_args()
    print("gen  args:", vars(args), "\n")
    placement_H = float(args.placement_H_L_delta[0])
    placement_L = float(args.placement_H_L_delta[1])
    placement_ker_delta = float(args.placement_H_L_delta[2])
    if not (placement_H > 0 and placement_L > 0):
        raise ValueError("placement_H_L_delta requires positive H and L")
    if placement_ker_delta <= 0:
        raise ValueError("placement_H_L_delta third value (placement_ker_delta) must be positive")

    placement_min_dist = float(args.placement_min_dist)
    if placement_min_dist <= 0:
        raise ValueError("placement_min_dist must be positive")

    synaptic_tau = float(args.synaptic_tau)
    std_tau_rec = float(args.std_recovery_tau)
    if synaptic_tau <= 0:
        raise ValueError("synaptic_tau must be positive")
    if std_tau_rec <= 0:
        raise ValueError("std_recovery_tau must be positive")
    if not (0 < args.std_u <= 1):
        raise ValueError("std_u must satisfy 0 < U <= 1")
    if args.logit_clip <= 0:
        raise ValueError("logit_clip must be positive")

    n_units = int(args.num_neurons)
    if n_units < 10:
        raise ValueError("num_neurons must be >= 10")
    if args.num_excite < 5 or args.num_excite >= n_units:
        raise ValueError("num_excite must satisfy 5 <= num_excite < num_neurons")
    if args.num_steps < 1000:
        raise ValueError("num_steps must be >= 1000")
    if args.step_size <= 0:
        raise ValueError("step_size must be positive")
    if args.idleRate[0] <= 0 or args.idleRate[1] <= args.idleRate[0]:
        raise ValueError("idleRate must satisfy 0 < min < max")
    if args.idleRate[1] * args.step_size >= 0.5:
        raise ValueError("idleRate[1] * step_size is too large for a Bernoulli approximation")

    total_time_sec = float(args.num_steps) * float(args.step_size)
    var_t_window = min(5.0, max(0.5, total_time_sec / 4.0))

    if args.kernel_len_steps < 1:
        raise ValueError("kernel_len_steps must be >= 1")
    mem_lag_steps = int(args.kernel_len_steps)
    kernel_span_sec = mem_lag_steps * float(args.step_size)
    kernel_capture_fraction = float(1.0 - np.exp(-kernel_span_sec / synaptic_tau))
    if args.verb > 0 and kernel_span_sec < 3.0 * synaptic_tau:
        print(
            "WARNING: kernel lag span M*dt=%.6g sec is less than 3*tau_s=%.6g sec; "
            "the fast synaptic tail will be strongly truncated."
            % (kernel_span_sec, 3.0 * synaptic_tau)
        )

    prob_lo = float(args.edge_prob[0])
    prob_hi = float(args.edge_prob[1])
    if not (0 < prob_lo < prob_hi <= 1.0):
        raise ValueError("edge_prob must satisfy 0 < lo < hi <= 1")
    k_out_min = max(1, int(np.floor(prob_lo * n_units)))
    k_out_max = int(np.floor(prob_hi * n_units))
    k_out_max = max(k_out_min, k_out_max)
    k_out_max = min(k_out_max, n_units - 1)

    print("gen BSSM-STD args:"); pprint( vars(args))

    rng = np.random.default_rng()
    out_path = os.path.join(args.basePath, "truthDale")
    if not os.path.exists(args.basePath):
        raise FileNotFoundError("missing basePath: %s" % args.basePath)
    if not os.path.exists(out_path):
        raise FileNotFoundError("missing output path: %s" % out_path)

    print("\n%s" % ("=" * 60))
    print("  Spectral radius target: R=%.3f" % float(args.spectral_radius))
    print("%s" % ("=" * 60))

    W_true, sign_true, positions, tau, rho0, distance_matrix = generate_spatial_dale_network(
        n_units=n_units,
        n_excite=args.num_excite,
        length_x=placement_L,
        height_y=placement_H,
        d_min=placement_min_dist,
        k_min=k_out_min,
        k_max=k_out_max,
        placement_ker_delta=placement_ker_delta,
        weight_var=args.init_weight_var,
        spectral_radius_target=args.spectral_radius,
        rng=rng,
        verb=args.verb,
    )

    if args.verb > 0:
        summarize_pairwise_distances(distance_matrix, verb=1)
        total_connections = W_true.size
        zero_connections = int(np.sum(np.abs(W_true) < 1e-10))
        non_zero_connections = total_connections - zero_connections
        sparsity = zero_connections / float(total_connections)
        print(
            "Matrix sparsity: %.1f%% (%d/%d connections are zero)"
            % (100.0 * sparsity, zero_connections, total_connections)
        )
        print("Non-zero connections: %d (%.1f%%)" % (non_zero_connections, 100.0 * (1.0 - sparsity)))

    kappa_true, synaptic_alpha = build_exponential_kernel(
        dt=float(args.step_size),
        synaptic_tau=synaptic_tau,
        mem_lag_steps=mem_lag_steps,
        verb=args.verb,
    )

    B_true = set_flat_selfSpiking(
        n_units=n_units,
        idle_rate_hz=args.idleRate,
        dt=float(args.step_size),
        tau=tau,
        rng=rng,
    )

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
        "init_weight_var": args.init_weight_var,
        "rho0_unscaled": rho0,
        "idleRate": args.idleRate,
        "model_name": "BSSM_STD",
        "mem_lag_steps": int(mem_lag_steps),
        "kernel_span_sec": float(kernel_span_sec),
        "kernel_capture_fraction": float(kernel_capture_fraction),
        "synaptic_tau": float(synaptic_tau),
        "std_tau_rec": float(std_tau_rec),
        "std_u": float(args.std_u),
        "self_loops": False,
    }
    evol_conf = {
        "num_steps": args.num_steps,
        "step_size": args.step_size,
        "evol_time": args.num_steps * args.step_size,
        "logit_clip": args.logit_clip,
        "model_name": "BSSM_STD",
        "mem_lag_steps": int(mem_lag_steps),
        "kernel_span_sec": float(kernel_span_sec),
    }

    print("\n%s" % ("=" * 60))
    print("  BSSM-STD Bernoulli spike generation")
    print("%s" % ("=" * 60))
    print("Dale configuration:")
    pprint(dale_conf)

    start_time = time.time()
    sim_out = gen_bssm_std_bernoulli(
        num_steps=args.num_steps,
        dt=float(args.step_size),
        W_true=W_true,
        B_true=B_true,
        tau=tau,
        std_u=float(args.std_u),
        std_tau_rec=float(std_tau_rec),
        synaptic_alpha=synaptic_alpha,
        mem_lag_steps=mem_lag_steps,
        logit_clip=float(args.logit_clip),
        rng=rng,
        verb=args.verb,
    )
    sim_time = time.time() - start_time
    print("Spike generation completed in %.1f seconds" % sim_time)

    spikes = sim_out["spikes"]
    max_samples = 100_000
    stats_dict, rates_dict, _ = estimate_rates(
        spikes,
        args.step_size,
        tau,
        max_samples,
        args.spectral_radius,
        varTwindow=var_t_window,
        mxNn=5,
        verb=0,
    )
    stats_dict["var_time_window_sec"] = float(var_t_window)
    stats_dict["max_samples"] = int(max_samples)

    trueD = {
        "W_true": W_true,
        "B_true": B_true,
        "sign_true": sign_true,
        "node_positions": positions,
        "node_is_inhibitory": tau,
        "node_distance_matrix": distance_matrix,
        "kappa_true": kappa_true.astype(np.float32),
        "x_final": sim_out["x_final"],
        "h_final": sim_out["h_final"],
        "x_true": sim_out["x_true"],
        "u_true": sim_out["u_true"],
        "h_true": sim_out["h_true"],
        "p_true": sim_out["p_true"],
    }

    max_spikes_per_neuron = spikes.max(axis=0)
    sp_min = float(max_spikes_per_neuron.min())
    sp_avg = float(max_spikes_per_neuron.mean())
    sp_max = float(max_spikes_per_neuron.max())
    sp_pc = np.percentile(max_spikes_per_neuron, [25, 50, 75]).tolist()

    if args.verb > 0:
        print("\nSpike count statistics (max per neuron over all bins):")
        print("  min: %.1f, avg: %.1f, max: %.1f" % (sp_min, sp_avg, sp_max))
        print("  percentiles [25, 50, 75]: %s" % sp_pc)

    trueMD = {
        "dale_conf": dale_conf,
        "evol_conf": evol_conf,
        "short_name": args.dataName,
        "provenance": {
            "state_model_file": args.dataName,
            "generator_script": os.path.basename(__file__),
        },
        "max_spike_stats": {
            "min": sp_min,
            "avg": sp_avg,
            "max": sp_max,
            "percentiles": sp_pc,
        },
    }

    spikeD = {
        "spikes": spikes.astype(np.uint8),
        "single_rates": rates_dict["single_rates"],
        "single_rates_var": rates_dict["single_rates_var"],
        "single_fano_fact": rates_dict["single_fano_fact"],
    }
    spikeMD = {
        "short_name": args.dataName,
        "time_step_sec": args.step_size,
        "data_type": "simBSSM_STD",
        "logit_clip": args.logit_clip,
        "model_name": "BSSM_STD",
        "placement_L": float(placement_L),
        "placement_H": float(placement_H),
        "placement_min_dist": float(placement_min_dist),
        "mem_lag_steps": int(mem_lag_steps),
        "kernel_span_sec": float(kernel_span_sec),
    }

    out_truth = os.path.join(out_path, args.dataName + ".simTruth.npz")
    write_data_npz(trueD, out_truth, metaD=trueMD)
    if args.verb > 1:
        pprint(trueMD)

    out_spikes = os.path.join(out_path, args.dataName + ".spikes.npz")
    write_data_npz(spikeD, out_spikes, metaD=spikeMD)
    if args.verb > 1:
        pprint(spikeMD)

    print("\nSimulation completed successfully!")
    print(
        "\nRate Summary %s  N=%d  exc=%d, R=%.3f"
        % (args.dataName, args.num_neurons, args.num_excite, float(args.spectral_radius))
    )
    s = stats_dict
    print(
        "  rates:  all=%14.1f  exc=%14.1f  inh=%14.1f"
        % (s["avg_spike_rate_all"], s["avg_spike_rate_excit"], s["avg_spike_rate_inhib"])
    )
    print("\nNext step commands:")
    print("     basePath=" + args.basePath)
    print(
        "  ./fit5_BSSM_STD_blocks.py --basePath $basePath --dataName %s --burn_sec 2 --init_samples 50000 --u_bounds 0.2 0.8 --tau_bounds 0.1 0.8"
        % args.dataName
    )
    print(
        "  ./eval_BSSM_fit.py --basePath $basePath --dataName %s -X"
        % args.dataName
    )


if __name__ == "__main__":
    main()
