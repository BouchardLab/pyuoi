#!/usr/bin/env python3
"""
Utilities for prism EM initialization.
"""

import numpy as np
import time


def _normalize_init_opt(opt):
    s = str(opt).strip().lower()
    if s not in ("data", "rand"):
        raise ValueError(f"Unsupported init option: {opt}")
    return s


def _rate_thresholds_min_var(rate_t, n_state):
    """Find up to (n_state-1) thresholds minimizing within-group variance."""
    x = np.asarray(rate_t, dtype=np.float64).ravel()
    if x.size == 0 or n_state <= 1:
        return []

    uniq, counts = np.unique(x, return_counts=True)
    u = uniq.size
    if u <= 1:
        return []

    k = min(int(n_state), u)
    w = counts.astype(np.float64)
    wx = w * uniq
    wx2 = wx * uniq

    c_w = np.concatenate(([0.0], np.cumsum(w)))
    c_s = np.concatenate(([0.0], np.cumsum(wx)))
    c_q = np.concatenate(([0.0], np.cumsum(wx2)))

    # dp[c, j]: minimal weighted SSE for first (j+1) unique values into c groups.
    dp = np.full((k + 1, u), np.inf, dtype=np.float64)
    ptr = np.full((k + 1, u), -1, dtype=np.int32)

    j_all = np.arange(u, dtype=np.int32) + 1
    w0 = c_w[j_all] - c_w[0]
    s0 = c_s[j_all] - c_s[0]
    q0 = c_q[j_all] - c_q[0]
    dp[1, :] = q0 - (s0 * s0) / np.maximum(w0, 1e-12)
    ptr[1, :] = 0

    for c in range(2, k + 1):
        for j in range(c - 1, u):
            i = np.arange(c - 1, j + 1, dtype=np.int32)
            jj = j + 1

            ww = c_w[jj] - c_w[i]
            ss = c_s[jj] - c_s[i]
            qq = c_q[jj] - c_q[i]
            seg = qq - (ss * ss) / np.maximum(ww, 1e-12)
            cand = dp[c - 1, i - 1] + seg

            ib = int(np.argmin(cand))
            dp[c, j] = float(cand[ib])
            ptr[c, j] = int(i[ib])

    starts = []
    j = u - 1
    for c in range(k, 1, -1):
        i = int(ptr[c, j])
        starts.append(i)
        j = i - 1
    starts.reverse()

    thres = []
    for i in starts:
        l = uniq[i - 1]
        r = uniq[i]
        thres.append(float(0.5 * (l + r)))
    return thres


def init_states_vs_time(spikes, dt, args):
    """Prepare initial c_hat probabilities and state labels from spikes.

    Returns:
      c_init: float32 array (T, M)
      S_init: int64 array (T,)
      init_meta: None or {"rate_thres": [..], "rate_bin_sec": float, "rate_bin_bins": int}
      freq_h1d: float32 array used for state classification
    """
    spikes = np.asarray(spikes)
    if spikes.ndim != 2:
        raise ValueError("spikes must be 2D array (T, N)")

    T_full = spikes.shape[0]
    n_state = int(args.num_states)
    opt = _normalize_init_opt(args.init_states)

    if opt == "rand":
        c_init = np.full((T_full, n_state), 1.0 / n_state, dtype=np.float32)
        s_init = np.full((T_full,), -1, dtype=np.int64)
        freq_h1d = np.zeros((0,), dtype=np.float32)
        return c_init, s_init, None, freq_h1d
    if opt != "data":
        raise ValueError(f"Unsupported --init_states option: {opt}")

    dwell_sec = float(args.decode_dwell_sec)
    dwell_bins = max(1, int(np.ceil(dwell_sec / float(dt))))

    # Aggregate spikes over coarse dwell bins, then compute mean firing rate.
    T = T_full
    n_blk = (T + dwell_bins - 1) // dwell_bins
    rate_blk = np.zeros((n_blk,), dtype=np.float64)
    blk_sizes = np.zeros((n_blk,), dtype=np.int64)
    for b in range(n_blk):
        i0 = b * dwell_bins
        i1 = min(T, i0 + dwell_bins)
        blk = spikes[i0:i1]
        nb = i1 - i0
        blk_sizes[b] = nb
        rate_per_neuron = blk.sum(axis=0).astype(np.float64) / (float(nb) * float(dt))
        rate_blk[b] = rate_per_neuron.mean()

    thres = _rate_thresholds_min_var(rate_blk, n_state)
    th = np.asarray(thres, dtype=np.float64)
    s_blk = np.searchsorted(th, rate_blk, side="right").astype(np.int64)
    s_blk = np.clip(s_blk, 0, max(0, n_state - 1))

    c_blk, s_blk_out = init_probs_from_states(s_blk, n_state)

    # Map coarse-bin initialization back to original binning.
    c_init = np.zeros((T_full, n_state), dtype=np.float32)
    s_init = np.zeros((T_full,), dtype=np.int64)
    for b in range(n_blk):
        i0 = b * dwell_bins
        i1 = i0 + int(blk_sizes[b])
        c_init[i0:i1] = c_blk[b]
        s_init[i0:i1] = s_blk_out[b]

    freq_h1d = rate_blk.astype(np.float32, copy=False)
    init_meta = {
        "rate_thres": [float(x) for x in thres],
        "rate_bin_sec": float(dwell_bins * float(dt)),
        "rate_bin_bins": int(dwell_bins),
    }
    return c_init, s_init, init_meta, freq_h1d


def init_selfspikingB(spikes, dt, args):
    """Initialize common per-state B from observed mean firing rates.

    For each neuron n:
      rate_hz[n] = mean_t spikes[t, n] / dt
      B[:, n] = log(max(rate_hz[n], eps))
    """
    spikes = np.asarray(spikes)
    if spikes.ndim != 2:
        raise ValueError("spikes must be 2D array (T, N)")

    n_state = int(args.num_states)
    eps_rate_hz = 1e-6

    rate_hz = spikes.mean(axis=0).astype(np.float64) / float(dt)
    rate_hz_clip = np.maximum(rate_hz, eps_rate_hz)
    b_vec = np.log(rate_hz_clip).astype(np.float32)
    b_init = np.tile(b_vec[None, :], (n_state, 1))

    init_meta = {
        "method": "rate",
        "common_across_states": True,
        "rate_floor_hz": float(eps_rate_hz),
        "formula": "B=log(max(rate_hz,eps))",
    }
    return b_init, init_meta


def init_B_from_spikes(spikes, dt, args):
    """Select B initialization mode based on args.init_B."""
    opt = _normalize_init_opt(args.init_B)
    if opt == "rand":
        return None, None
    if opt != "data":
        raise ValueError(f"Unsupported --init_B option: {opt}")
    return init_selfspikingB(spikes, dt, args)


def init_edgesA(spikes, Tmax=50000, verbose=True):
    """OLS/covariance initialization of A from spike data Y (T x N)."""
    t0 = time.perf_counter()
    Y = np.asarray(spikes, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("spikes must be 2D array (T, N)")
    T_full, _ = Y.shape
    T_use = min(int(Tmax), int(T_full))
    if T_use < 2:
        raise ValueError("Need at least 2 time bins to initialize A")
    Y = Y[:T_use]

    YpYp = Y[:-1].T @ Y[:-1]   # Gram matrix
    YYp = Y[1:].T @ Y[:-1]     # one-step cross-correlation

    A_ols = YYp @ np.linalg.pinv(YpYp)

    kappa = float(np.linalg.cond(YpYp))
    rho = float(np.max(np.abs(np.linalg.eigvals(A_ols))))
    Y_pred = Y[:-1] @ A_ols.T
    var_y = float(np.var(Y[1:]))
    if var_y <= 0.0:
        R2 = float("nan")
    else:
        R2 = float(1.0 - np.var(Y[1:] - Y_pred) / var_y)
    frob = float(np.linalg.norm(A_ols, ord="fro"))
    elapsed_sec = float(time.perf_counter() - t0)

    if verbose:
        print("A-init OLS diagnostics:")
        print(f"  cond(YpYp) = {kappa:.2e}")
        print(f"  rho(A_ols) = {rho:.3f}")
        print(f"  R2         = {R2:.3f}")
        print(f"  ||A||_F    = {frob:.3f}")
        print(f"  bins_used  = {T_use}/{T_full}")
        print(f"  elapsed_s  = {elapsed_sec:.3f}")

    meta = {
        "method": "cov_ols",
        "Tmax": int(Tmax),
        "num_bins_used": int(T_use),
        "num_bins_total": int(T_full),
        "cond_YpYp": kappa,
        "rho_A_init": rho,
        "R2_1step": R2,
        "fro_A_init": frob,
        "elapsed_init_edgesA_sec": elapsed_sec,
    }
    return A_ols.astype(np.float32), meta


def init_A_from_spikes(spikes, args):
    """Select A initialization mode based on args.init_A."""

    opt = _normalize_init_opt(args.init_A)
    if opt == "rand":
        return None, None
    if opt != "data":
        raise ValueError(f"Unsupported --init_A option: {opt}")
    A_init, meta = init_edgesA(spikes, verbose=False)

    if 1:  # rescale A 
        offDiagFact = 3
        meta["offDiag_A_init_fact"] = offDiagFact
        A_init = A_init * offDiagFact

    if 0:  # thresholded off-diagonal renormalization        
        off_diag = ~np.eye(A_init.shape[0], A_init.shape[1], dtype=bool)
        above_thr = (np.abs(A_init) > float(args.minW))
        below_thr = (np.abs(A_init) < float(args.minW))
        strong_off_diag = off_diag & above_thr
        weak_off_diag = off_diag & below_thr
        A_init[strong_off_diag] *= offDiagFact
        A_init[weak_off_diag] = 0.0
        

    rho_max = float(args.rho_max)
    if 0: # global spectral radius rescaling
        rho_before = float(np.max(np.abs(np.linalg.eigvals(A_init))))
        if rho_before > rho_max:
            A_init = A_init * (rho_max / rho_before)
        rho_after = float(np.max(np.abs(np.linalg.eigvals(A_init)))) 
        meta["rho_A_init_raw"] = rho_before
        meta["rho_A_init"] = rho_after
    meta["rho_max_target"] = rho_max

    if args.verb > 0:
        print("A-init OLS diagnostics:")
        print(f"  cond(YpYp) = {meta['cond_YpYp']:.1f}")
        print(f"  rho(A_ols) = {meta['rho_A_init']:.3f}")
        print(f"  R2         = {meta['R2_1step']:.3f}")
        print(f"  ||A||_F    = {meta['fro_A_init']:.3f}")
        print(f"  bins_used  = {meta['num_bins_used']}/{meta['num_bins_total']}")
        print(f"  elapsed_s  = {meta['elapsed_init_edgesA_sec']:.3f}")

    return A_init, meta


def init_probs_from_states(S_rate, M):
    """Build c_init from S_rate with transition smoothing."""
    S_rate = np.asarray(S_rate, dtype=np.int64)
    T = S_rate.size
    if T == 0:
        return np.zeros((0, M), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if M <= 1:
        return np.ones((T, 1), dtype=np.float32), S_rate.copy()

    # Base: dominant state at 0.8, remaining 0.2 spread over all others.
    other = 0.2 / float(M - 1)
    c_init = np.full((T, M), other, dtype=np.float32)
    c_init[np.arange(T), S_rate] = 0.8
    S_out = S_rate.copy()

    # Transition smoothing: (t-1) gets 2/3 old + 1/3 new, t gets 1/3 old + 2/3 new.
    tr_idx = np.where(S_rate[1:] != S_rate[:-1])[0] + 1
    for t in tr_idx:
        a = int(S_rate[t - 1])
        b = int(S_rate[t])
        c_init[t - 1, :] = 0.0
        c_init[t, :] = 0.0
        c_init[t - 1, a] = 2.0 / 3.0
        c_init[t - 1, b] = 1.0 / 3.0
        c_init[t, a] = 1.0 / 3.0
        c_init[t, b] = 2.0 / 3.0

    return c_init, S_out
