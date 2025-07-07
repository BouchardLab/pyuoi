#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__  = "janstar1122@gmail.com"

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time
import numpy as np

def simulate_cox_ou_correlated(dt_ms,
                               lambda_rate,
                               nFeat,
                               tMax_s,
                               tau_c_ms=30,
                               mod_depth=1.0,
                               corr_strength=0.5,
                               rng=None):
    """
    Simulate nFeat spike‐trains whose instantaneous rate on channel i is
        λ_i(t) = lambda_rate + x0(t) + x_i(t)
    where
      x0(t)   = shared OU‐process (time‐const = tau_c_ms)
      x_i(t)  = private OU‐process for channel i
    The total stationary variance of x0+x_i is (mod_depth*lambda_rate)^2,
    and corr_strength ∈ [0,1] determines what fraction of that variance
    lives in the shared component x0.

    Inputs:
      dt_ms        : time‐bin in ms
      lambda_rate  : baseline rate in spikes/sec
      nFeat        : number of independent channels
      tMax_s       : total simulation time in seconds
      tau_c_ms     : OU time‐constant in ms
      mod_depth    : relative total std‐dev of rate fluctuations (std(x)/lambda_rate)
      corr_strength: fraction of var(x) that is shared (0→indep,1→all shared)
      rng          : Optional np.random.Generator

    Returns:
      spikes       : boolean array (nFeat, nTime) of 0/1 spike‐events
    """

    if rng is None:
        rng = np.random.default_rng()

    # convert units
    dt_s   = dt_ms / 1000.0
    tau_c  = tau_c_ms/1000.0
    nTime  = int(round(tMax_s * 1000.0 / dt_ms))

    # desired variances
    var_tot    = (mod_depth * lambda_rate)**2
    var_common = var_tot * corr_strength
    var_priv   = var_tot * (1.0 - corr_strength)

    # OU noise amplitudes: Var(x)=σ² τc/2 ⇒ σ = sqrt(2 Var / τc)
    sigma_c = np.sqrt(2.0 * var_common / tau_c)
    sigma_p = np.sqrt(2.0 * var_priv   / tau_c)

    # simulate shared x0(t) and private x_i(t) by Euler–Maruyama
    x0 = np.zeros(nTime, dtype=float)
    xP = np.zeros((nFeat, nTime), dtype=float)

    for t in range(1, nTime):
        # shared OU update
        x0[t] = x0[t-1] + (-x0[t-1]/tau_c)*dt_s \
                + sigma_c * np.sqrt(dt_s) * rng.standard_normal()
        # private OU update (vectorized over channels)
        xP[:,t] = xP[:,t-1] + (-xP[:,t-1]/tau_c)*dt_s \
                  + sigma_p * np.sqrt(dt_s) * rng.standard_normal(nFeat)

    # instantaneous rate matrix
    lam = lambda_rate + x0[None,:] + xP
    lam[lam < 0] = 0.0

    # Poisson sampling (gives counts, we threshold to bits)
    counts = rng.poisson(lam * dt_s, size=(nFeat, nTime))
    spikes = counts > 0

    return spikes


def fit_exp_weighted(times,
                     C,
                     dt_ms,
                     N_total,
                     fit_start_ms=5):
    """
    Weighted fit of C(t) = A * exp(-t/tau) for t >= fit_start_ms.
    times        : array of lags (s)
    C            : array of covariances
    dt_ms        : bin‐size in ms
    N_total      : total # bins per channel used in compute_cov
    fit_start_ms : ignore lags < this (ms)
    returns (tau, A, tau_err)
    """

    # 1) figure out which lags we’re fitting
    start_i = int(np.ceil(fit_start_ms / dt_ms))
    start_i = max(start_i, 1)    # never use the 0‐lag

    t_all = times[start_i:]
    C_all = C[start_i:]
    ks    = np.arange(start_i, start_i + len(C_all))

    # 2) throw away any negative C (they can't be fit by positive‐A exponential)
    pos    = C_all > 0
    t_fit  = t_all[pos]
    C_fit  = C_all[pos]
    ks_fit = ks[pos]

    # 3) build a sensible positive initial guess for A and tau
    A0   = C_fit[0]
    if A0 <= 0:
        A0 = C_fit.max()
    tau0 = (t_fit[np.argmin(np.abs(C_fit - C_fit[0]/np.e))]
            if np.any(C_fit < C_fit[0]/np.e)
            else t_fit[-1])
    p0 = (A0, tau0)

    # 4) weights ~ 1/sqrt(N_total - k)
    sigma = 1.0/np.sqrt(N_total - ks_fit)

    # 5) do the curve‐fit with A>=0, tau>=0
    def model(t, A, tau):
        return A * np.exp(-t/tau)

    lower = (0.0, 0.0)
    upper = (np.inf, np.inf)
    popt, pcov = curve_fit(model,
                           t_fit, C_fit,
                           p0=p0,
                           sigma=sigma,
                           absolute_sigma=False,
                           bounds=(lower, upper))
    A_est, tau_est = popt
    tau_err = np.sqrt(np.diag(pcov))[1]

    return tau_est, A_est, tau_err


def compute_fano(spikes, dt_ms, windows_ms):
    """
    Bin into non‐overlapping windows and compute mean, var, fano.
    """
    nFeat, nTime = spikes.shape
    results = []
    for w in windows_ms:
        bs   = max(1, int(round(w / dt_ms)))
        nBins = nTime // bs
        data  = spikes[:, :nBins*bs]
        counts = data.reshape(nFeat, nBins, bs).sum(axis=2)
        flat   = counts.ravel().astype(float)
        m = flat.mean()
        v = flat.var(ddof=0)
        f = v/m if m>0 else np.nan
        results.append((w, m, v, f))
    return zip(*results)

def print_count_fano_table(windows_ms, means, variances, fano_factors):
    h = f"{'Window(ms)':>10s}  {'Mean':>10s}  {'Variance':>10s}  {'Fano':>8s}"
    print(h); print('-'*len(h))
    for w, m, v, f in zip(windows_ms, means, variances, fano_factors):
        print(f"{w:10d}  {m:10.3f}  {v:10.3f}  {f:8.3f}")

def compute_mean_crosscov_fast(spikes, dt_ms, max_lag_ms=None):
    """
    Fast approximation to the average cross‐covariance over all i<j.
    Ignores the small 'self' term, which for nFeat~400 gives <1% bias.

    spikes    : bool or {0,1} array, shape (nFeat, nTime)
    dt_ms     : bin size in ms
    max_lag_ms: maximum lag to compute (in ms); if None uses full record

    Returns
      times : array of lags [s], length L
      C     : array of approximate cross‐covariances, length L
    """
    nFeat, N = spikes.shape
    dt  = dt_ms/1000.0
    if max_lag_ms is None:
        max_lag = N-1
    else:
        max_lag = min(int(max_lag_ms/dt_ms), N-1)

    # 1) zero‐mean each channel
    S = spikes.astype(np.float64)
    S -= S.mean(axis=1, keepdims=True)

    # 2) form the sum over channels
    R = S.sum(axis=0)   # length N

    # 3) number of distinct pairs
    nPairs = nFeat*(nFeat-1)/2

    # 4) allocate output
    L = max_lag+1
    C = np.empty(L, dtype=np.float64)

    # 5) for each lag k, do one dot() of two length-(N–k) vectors
    for k in range(L):
        Nk = N - k
        C[k] = R[:Nk].dot(R[k:]) / (Nk * nPairs)

    # 6) time‐axis
    times = np.arange(L) * dt
    return times, C

def compute_mean_crosscov(spikes, dt_ms, max_lag_ms=None):
    """
    Compute unbiased cross‐covariance averaged over i<j pairs.
    """
    nFeat, N = spikes.shape
    dt = dt_ms/1000.0
    max_lag = N-1 if max_lag_ms is None else min(int(max_lag_ms/dt_ms), N-1)

    # zero‐mean each channel
    S = spikes.astype(float)
    S0 = S - S.mean(axis=1, keepdims=True)

    # accumulate over all pairs
    Csum = np.zeros(max_lag+1, dtype=float)
    nPairs = 0
    for i in range(nFeat):
        xi = S0[i]
        for j in range(i+1, nFeat):
            xj = S0[j]
            for k in range(max_lag+1):
                Csum[k] += np.dot(xi[:N-k], xj[k:])/(N-k)
            nPairs += 1

    C = Csum / nPairs
    times = np.arange(max_lag+1) * dt
    return times, C

def plot_crosscov_with_fit(ax,
                           times, C,
                           dt_ms,
                           tau_c, A, tau_err,
                           fit_start_ms=5,
                           filename="crosscov_fit.png"):
    """
    Plot times,C and overplot the exp‐fit for t>=fit_start_ms.
    Save to filename and print it.
    """
    # index to start fit‐curve
    start_i = max(int(np.ceil(fit_start_ms/dt_ms)), 1)
    t_fit   = times[start_i:]
    c_fit   = A * np.exp(-t_fit / tau_c)

    # plot
    ax.plot(times[1:], C[1:], 'k.', label='empirical')
    ax.plot(t_fit,      c_fit,  'r-', label=f'fit τc={tau_c:.4f}±{tau_err:.4f}s')
    ax.axvline(fit_start_ms/1000.0, color='gray', linestyle=':',
               label=f'start at {fit_start_ms}ms')

    ax.set_xlabel('lag (s)')
    ax.set_ylabel('cross‐covariance')
    ax.set_title('cross‐covariance')
    ax.legend(loc='best')
    ax.grid(True)

    # save
    fig = ax.get_figure()
    od  = os.path.dirname(filename)
    if od and not os.path.exists(od):
        os.makedirs(od)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")

def main():
    # simulation params
    dt_ms    = 1       # ms
    lambda_rate      = 50      # spikes/sec
    nFeat    = 102     # number of channels
    tMax_s   = 301     # seconds
    
    tau_c_ms = 40      # true OU tau in ms    
    mod_depth= 0.5       # ±50% relative fluctuations
    corr_str = 0.3       # 30% of the variance is shared

    spikes = simulate_cox_ou_correlated(dt_ms, lambda_rate, nFeat, tMax_s,
                                    tau_c_ms, mod_depth, corr_str)


    # simulate
    #spikes = simulate_cox_ou(dt_ms, tau, nFeat, tMax_s, tau_c_ms, depth)
    print("Sample size (nFeat, nTime):", spikes.shape)

    # Fano‐factor vs window
    windows_ms = [2**k for k in range(0, 11)]
    ws, ms, vs, fs = compute_fano(spikes, dt_ms, windows_ms)
    print_count_fano_table(ws, ms, vs, fs)

    # cross‐covariance
    T0=time()
    times, C = compute_mean_crosscov_fast(spikes, dt_ms, max_lag_ms=2*tau_c_ms)
    print('M: %s cross-cov computed in elaT=%.1f sec'%(str(spikes.shape),time() -T0))
    fit_start = 5   # ms
    N_total   = spikes.shape[1]
    tau_c, A, tau_err = fit_exp_weighted(times, C, dt_ms, N_total, fit_start)
    print(f"Estimated tau_c = {tau_c:.3f} ± {tau_err:.3e} s, A = {A:.3e}")

    # plot
    fig, ax = plt.subplots(figsize=(6,4))
    plot_crosscov_with_fit(ax, times, C, dt_ms,
                           tau_c, A, tau_err,
                           fit_start_ms=fit_start,
                           filename="crosscov_fit.png")
    plt.show()

if __name__ == "__main__":
    main()
