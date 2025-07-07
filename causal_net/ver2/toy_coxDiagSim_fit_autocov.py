#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
compute_mean_autocov does an unbiased lag‐k covariance
C[k] = 1/(N−k) ∑ᵢ(sᵢ[t]−μᵢ)(sᵢ[t+k]−μᵢ), averaged over features i.
fit_exponential_decay then fits C(τ)∼A·exp(−τ/τc) for τ above a small cutoff to avoid the shot‐noise “spike” at very small lags.
The returned tau_c is your estimated correlation time (in seconds).
'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def compute_mean_autocov(spikes, dt_ms, max_lag_ms=None):
    """
    Compute unbiased autocovariance C[k] = Cov[s[t], s[t+k]] averaged over channels.
    spikes     : bool or {0,1} array of shape (nFeat, nTime)
    dt_ms      : time‐bin size in ms
    max_lag_ms : maximum lag to compute (in ms); if None uses full record
    Returns:
      times : array of lags (seconds), length L
      C     : array of autocovariances, same length L
    """
    nFeat, N = spikes.shape
    dt = dt_ms/1000.0
    if max_lag_ms is None:
        max_lag = N-1
    else:
        max_lag = min(int(max_lag_ms/dt_ms), N-1)
    Csum = np.zeros(max_lag+1, dtype=float)

    for i in range(nFeat):
        s = spikes[i].astype(float)
        μ = s.mean()
        s0 = s - μ
        # unbiased autocov for lags 0..max_lag
        for k in range(max_lag+1):
            Csum[k] += np.dot(s0[:N-k], s0[k:])/(N-k)

    C = Csum / nFeat
    times = np.arange(max_lag+1) * dt
    return times, C



def fit_exp_weighted(times, C, dt_ms,N_total, fit_start_ms=5):
    # select region
    dt_s    = dt_ms/1000.0
    start_i = int(np.ceil(fit_start_ms/(dt_ms)))
    t_fit   = times[start_i:]
    C_fit   = C[start_i:]
    L       = len(C_fit)
    #N_total    # total bins used in compute_mean_autocov
    # approximate std(C[k]) ≃ 1/√(N_total - k)
    ks      = np.arange(start_i, start_i+L)
    sigma   = 1.0/np.sqrt(N_total - ks)  

    # model
    def model(t, A, tau):
        return A*np.exp(-t/tau)

    p0 = (C_fit[0], 0.1)   # initial guess
    popt, pcov = curve_fit(model,
                           t_fit, C_fit,
                           p0=p0,
                           sigma=sigma,
                           absolute_sigma=True,
                           bounds=(0, np.inf))
    A_est, tau_est = popt
    perr = np.sqrt(np.diag(pcov))
    return tau_est, A_est, perr[1]   # return τc, A and stderr of τc

def simulate_cox_ou(dt, tau, nFeat, tMax, tau_c=30, mod_depth=1.0, rng=None):
    """
    Simulate nFeat independent spike‐trains whose rate λ_i(t) is
      λ_i(t) = tau + x_i(t)
    with x_i an OU process of time‐constant tau_c (in seconds)
    and stationary std dev = mod_depth * tau.
    
    dt        : bin size in ms
    tau       : baseline rate in spikes/sec
    nFeat     : number of independent channels
    tMax      : total time in seconds
    tau_c     : OU time constant in ms 
    mod_depth : relative fluctuation size = std(x)/tau
    rng       : numpy Generator (optional)
    
    “Cox‐process’’ generator in which each channel’s instantaneous rate is a random Ornstein–Uhlenbeck (OU) process with correlation time τc

    Returns:
      spikes : bool array of shape (nFeat, nTime)
               True = spike in that dt‐bin
    """
    if rng is None:
        rng = np.random.default_rng()

    dt_s   = dt / 1000.0
    tau_c/=1000.
    # now time is in seconds
    nTime  = int(np.round(tMax * 1000.0 / dt))
    
    # OU‐process parameters
    #   dx = −(x/τc) dt + σ dW  ⇒  Var(x) = σ² τc/2
    # so to get std(x)=mod_depth*tau, choose
    sigma = mod_depth * tau * np.sqrt(2.0 / tau_c)
    
    # pre‐allocate
    x = np.zeros((nFeat, nTime), dtype=float)
    
    # simulate OU by Euler‐Maruyama
    for t in range(1, nTime):
        x[:, t] = x[:, t-1] \
                  + ( - x[:, t-1] / tau_c ) * dt_s \
                  + sigma * np.sqrt(dt_s) * rng.standard_normal(nFeat)
    
    # instantaneous rates (clipped ≥0)
    lam = tau + x
    lam[lam < 0] = 0.0

    # draw Poisson counts in each bin, then threshold to bits
    counts = rng.poisson(lam * dt_s, size=(nFeat, nTime))
    spikes = counts > 0
    return spikes

def compute_fano(spikes, dt, windows_ms):
    """
    For each window size w in windows_ms, bin spikes into non-overlapping
    windows of length w, compute:
      mean count per bin, variance per bin, Fano=var/mean
    spikes  : bool array (nFeat, nTime)
    dt      : time step in ms
    windows_ms : iterable of integer window sizes in ms
    Returns:
      four lists: windows_ms, means, variances, fano_factors
    """
    nFeat, nTime = spikes.shape
    results = []
    for w in windows_ms:
        bin_size = int(round(w / dt))
        if bin_size < 1:
            bin_size = 1
        nBins = nTime // bin_size
        # truncate to an integer number of bins
        data = spikes[:, : nBins*bin_size]
        # reshape to (nFeat, nBins, bin_size) and sum over the last axis
        counts = data.reshape(nFeat, nBins, bin_size).sum(axis=2)
        flat = counts.ravel().astype(float)
        m = flat.mean()
        v = flat.var(ddof=0)
        f = v / m if m>0 else np.nan
        results.append((w, m, v, f))
    # unzip
    ws, ms, vs, fs = zip(*results)
    return ws, ms, vs, fs


def print_count_fano_table(windows_ms, means, variances, fano_factors):
    """
    Print a table of:
      Window(ms) | mean spike‐count | variance of count | Fano factor
    """
    header = f"{'Window(ms)':>10s}  {'Mean':>10s}  {'Variance':>10s}  {'Fano':>8s}"
    print(header)
    print('-' * len(header))
    for w, m, v, f in zip(windows_ms, means, variances, fano_factors):
        print(f"{w:10d}  {m:10.3f}  {v:10.3f}  {f:8.3f}")

def plot_autocov_with_fit(ax,
                          times,
                          C,
                          dt_ms,
                          tau_c,
                          A,
                          tau_err,
                          fit_start_ms=5,
                          filename="autocov_fit.png"):
    """
    ax           : a matplotlib Axes
    times        : 1D array of lags (in seconds)
    C            : 1D array of autocovariances
    dt_ms        : bin‐size in ms (e.g. 1.0)
    tau_c, A     : fitted parameters of C ≈ A·exp(–t/tau_c)
    tau_err      : standard error of tau_c
    fit_start_ms : start fitting at this lag (ms)
    filename     : path to save the PNG
    """

    # 1) compute index of first lag ≥ fit_start_ms
    start_idx = int(np.ceil(fit_start_ms / dt_ms))
    start_idx = max(start_idx, 1)

    # 2) build fit curve only in fitted region
    t_fit = times[start_idx:]
    c_fit = A * np.exp(-t_fit / tau_c)

    # 3) plot empirical data (skip lag=0)
    ax.plot(times[1:], C[1:], 'k.', label='empirical')

    # 4) plot fit
    ax.plot(t_fit,
            c_fit,
            'r-',
            label=f'fit τc={tau_c:.3f}±{tau_err:.3f}s')

    # 5) vertical line showing start of fit
    ax.axvline(fit_start_ms/1000.0,
               color='gray',
               linestyle=':',
               label=f'start at {fit_start_ms}ms')

    # 6) decorate
    ax.set_xlabel('lag (s)')
    ax.set_ylabel('autocovariance')
    ax.set_title('autocovariance')
    ax.legend(loc='best')
    ax.grid(True)

    
    # save figure
    fig = ax.get_figure()
    # ensure directory exists
    outdir = os.path.dirname(filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.show()
    
# -----------------------------------------------------------------------
# Example of use in your main script:

if __name__=='__main__':
    # suppose `spikes` is your (nFeat, nTime) boolean array
    # and dt_ms is your sampling bin in ms
   
    # spikes = ... load or simulate ...
    # Simulation parameters
    dt_ms     = 1       # ms
    tau    = 50      # spikes/sec
    nFeat  =  102      # number of independent channels
    tMax   = 601     # seconds
    
    # Cox‐process with Ornstein–Uhlenbeck (OU) process with correlation time τc
    tau_c  = 40    #  ms correlation time
    depth  = 0.3     #  relative SD of rate fluctuations

    spikes = simulate_cox_ou(dt_ms, tau, nFeat, tMax, tau_c, depth)
    print('M: sample size:',spikes.shape)
    
    # Build window list 1,2,4,...,1024 ms
    windows_ms = [2**k for k in range(0, 11)]

    # Compute Fano‐factor vs window
    ws, ms, vs, fs = compute_fano(spikes, dt_ms, windows_ms)

    print_count_fano_table(ws, ms, vs, fs)
    
    
    # 1) compute avg autocov up to, say, 200 ms
    times, C = compute_mean_autocov(spikes, dt_ms, max_lag_ms=50)

    # get weighted fit + error
    fit_start=5
    N_total=spikes.shape[1]
    tau_c, A, tau_err = fit_exp_weighted(times, C, dt_ms, N_total, fit_start_ms= fit_start)

    print("Estimated tau_c = %.3f +/- %.3f , A=%.3e"%(tau_c,tau_err,A))

    # Optionally plot
    fig, ax = plt.subplots(figsize=(6,4))
    fig, ax = plt.subplots(figsize=(6,4))
    plot_autocov_with_fit(ax,
                      times,
                      C,
                      dt_ms=1.0,
                      tau_c=tau_c,
                      A=A,
                      tau_err=tau_err,
                      fit_start_ms=5,
                      filename="autocov_fit.png")
    
    #plot_autocov_with_fit(ax, times,  C, tau_c, A, fit_start_ms=fit_start_ms,
    #                      filename="autocov_vs_lag.png")
   
