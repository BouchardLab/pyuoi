#!/usr/bin/env python3

import numpy as np
from scipy.optimize import curve_fit

#...!...!.................... 
def print_dale_matrix(A,nfeat=None):
    if nfeat==None: nfeat=A.shape[0]
    # Function to format values
    def format_value(val):
        if abs(val) < 0.01:
            return "  .  "  # Represent zero as '-'
        return f"{val:+5.2f}"  # Format as +0.12 or -0.23
    
    col_indices = "feat " + "     ".join(f"{i:2d}" for i in range(nfeat))
    print(col_indices)
    # Print row index and formatted values
    for i in range(nfeat):
        row=A[i]
        formatted_row = "  ".join(format_value(row[j]) for j in range(nfeat) )
        print(f"{i:2d}  {formatted_row}")  # Row index + formatted values

    
#...!...!.................... 
def rebin_axis0_average(V, k):
    """
    Average‑rebin along axis 0 by factor k.
    If the length along axis 0 is not divisible by k, the input is clipped
    (extra samples at the end are dropped).
    
    Parameters
    ----------
    V : array‑like, shape (nt, ...)
        Input data.
    k : int
        Rebin factor.
    
    Returns
    -------
    rebinned : ndarray, shape (nt//k, ...)
        Data averaged over non‑overlapping blocks of size k along axis 0.
    """
    nt = V.shape[0]
    # drop extra samples so length is divisible by k
    trimmed_len = nt - (nt % k)
    if trimmed_len != nt:
        V = V[:trimmed_len]
    new_shape = (trimmed_len // k, k) + V.shape[1:]
    return V.reshape(new_shape).mean(axis=1)


#...!...!.................... 
def daleMatrix_index_partition(C):
    """
    Given a square matrix C of shape (2N, 2N) (list-of-lists or ndarray),
    return five index‐tuples for NumPy advanced indexing:
      Ldia   – main diagonal
      Lexc   – first‐N columns, off‐diagonal nonzeros
      Lzexc  – first‐N columns, off‐diagonal zeros
      Linh   – last‐N  columns, off‐diagonal nonzeros
      Lzinh  – last‐N  columns, off‐diagonal zeros

    Each is a tuple (rows, cols), so you can do C[rows, cols] to extract them.
    """
    C = np.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square 2D array")
    n2 = C.shape[0]
    if n2 % 2:
        raise ValueError("Dimension must be even (2N x 2N)")
    N = n2 // 2

    # 1) diagonal
    Ldia = np.diag_indices(n2)

    # 2) build a full grid of indices
    rows, cols = np.indices((n2, n2))

    # 3) masks for E‐region (cols< N) and I‐region (cols>=N), excluding diag
    maskE = (cols <  N) & (rows != cols)
    maskI = (cols >= N) & (rows != cols)

    # 4) split each region into nonzero / zero
    Lexc  = np.where(maskE & (C != 0))
    Lzexc = np.where(maskE & (C == 0))
    Linh  = np.where(maskI & (C != 0))
    Lzinh = np.where(maskI & (C == 0))

    return Ldia, Lexc, Lzexc, Linh, Lzinh

#...!...!.................... 

def residual_stats(A: np.ndarray):
    """
    Compute
      • mean of X  (μ_X)
      • mean of Y  (μ_Y)
      • Pearson correlation ρ(X,Y)
      • demeaned-and-rotated data  (X′, Y′)

    The returned (X′, Y′) satisfy
      – their means are zero, and
      – the horizontal axis (X′) is the direction of maximal variance,
        so the vertical variance (along Y′) is minimal.

    Parameters
    ----------
    A : numpy.ndarray, shape (n,2)
        Input data; column 0 → X, column 1 → Y.

    Returns
    -------
    stats : dict with keys ('mu_X','mu_Y','rho','theta')
    X_prime, Y_prime : 1-D numpy arrays of length n
    """
    # 0. check shape and split into X,Y
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("Input must be an (n,2) array")
    X = A[:, 0].astype(float)
    Y = A[:, 1].astype(float)
    N = X.shape[0]

    # 1. basic statistics
    mu_X = X.mean()
    mu_Y = Y.mean()
    rho  = np.corrcoef(X, Y)[0, 1]

    # 2. centre the data
    #    Stack as 2×n so we can re-use your rotation code unchanged
    Xc = np.stack([X - mu_X, Y - mu_Y], axis=0)  # shape (2, n)

    # 3. compute covariance and principal axis
    C      = np.cov(Xc)                          # 2×2
    eigv, eigvecs = np.linalg.eigh(C)            # ascending eigenvalues
    v_max  = eigvecs[:, np.argmax(eigv)]         # direction of max variance
    theta  = np.arctan2(v_max[1], v_max[0])

    # 4. rotate so that v_max aligns with horizontal axis
    R      = np.array([[ np.cos(-theta), -np.sin(-theta)],
                       [ np.sin(-theta),  np.cos(-theta)]])
    Xr     = R @ Xc                              # shape (2, n)
    X_prime, Y_prime = Xr                        # unpack back into two 1-D arrays

    stdX=np.std(X_prime)
    stdY=np.std(Y_prime)
    # standard error of the std estimator is :  se_s = std / np.sqrt(2 * (N - 1))
    statsD = {
        'mu_X':  mu_X,
        'mu_Y':  mu_Y,
        'res_std': np.std(X-Y),
        'rho':   rho,
        'theta': theta,
        'std_Xp':stdX,
        'stdE_Xp':stdX/np.sqrt(2 * (N - 1)), 
        'std_Yp': stdY,
        'stdE_Yp':stdY/np.sqrt(2 * (N - 1)), 
    }
    return statsD, X_prime, Y_prime






#...!...!.................... 

def compute_spike_moments(spikes, maxRebin=10,maxTime=300_000,verb=1):
    """
    For each window size w in windows_ms, bin spikes into non-overlapping
    windows of length w, compute:
      mean count per bin, variance per bin, Fano=var/mean
    spikes  : bool array (nFeat, nTime)
  
    windows_ms : iterable of integer window sizes in  bins
    Returns:
      four lists: windows_ms, means, variances, fano_factors
    """
    windows= [2**k for k in range(0, maxRebin)]
    
    nFeat, nTime = spikes.shape
    nTime =min(nTime,maxTime)
    print('compute_spike_moments spikes(%d,%d) windows:'%(nFeat, nTime), windows)
    
    results = []
    for bin_size in windows: # w is bin_size
        nBins = nTime // bin_size
        # truncate to an integer number of bins
        data = spikes[:, : nBins*bin_size]
        # reshape to (nFeat, nBins, bin_size) and sum over the last axis
        counts = data.reshape(nFeat, nBins, bin_size).sum(axis=2)
        flat = counts.ravel().astype(float)
        m = flat.mean()
        v = flat.var(ddof=0)
        f = v / m if m>0 else np.nan
        results.append((bin_size, m, v, f))
        #print('done',bin_size, m, v, f)

    if verb>0:    # unzip
        ws, ms, vs, fs = zip(*results)
        header = f"{'Window ':>10s}  {'Spikes Mean':>10s}  {'Variance':>10s}  {'Fano fact':>8s}"
        print(header)
        print('-' * len(header))
        for w, m, v, f in results:
            #if w not in [ 1,16,128,1024]: continue
            print(f"{w:10d}  {m:10.3f}  {v:10.3f}  {f:8.3f}")

    return np.array(results)
 



#...!...!.................... 
def fit_exponent_weighted(times,
                     C,
                     dt_ms,
                     N_total,
                     fit_start_ms=5):
    """
    Weighted fit of C(t) = A * exp(-t/tau) + B for t >= fit_start_ms.
    times        : array of lags (s)
    C            : array of covariances
    dt_ms        : bin‐size in ms
    N_total      : total # bins per channel used in compute_cov
    fit_start_ms : ignore lags < this (ms)
    returns dictionary with fitted parameters and fitting range
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
    p0 = (A0, tau0, 0)
    # 4) weights ~ 1/sqrt(N_total - k)
    sigma = 1.0/np.sqrt(N_total - ks_fit)

    # 5) do the curve‐fit with A>=0, tau>=0
    def model(t, A, tau, B):
        return A * np.exp(-t/tau)+B
        #yA=A * np.exp(-t/tau)
        #return np.sqrt(yA**2+B**2)
    lower = (0.0, 0.0, 0.0)
    upper = (np.inf, np.inf, np.inf)
    popt, pcov = curve_fit(model,
                           t_fit, C_fit,
                           p0=p0,
                           sigma=sigma,
                           absolute_sigma=not False,
                           bounds=(lower, upper))
    A_est, tau_est, B_est = popt
    perr = np.sqrt(np.diag(pcov))
    A_err, tau_err, B_err = perr
    print('tau_est=%.3f (s)   A_est=%.2e     B_est=%.2e    '%(tau_est,A_est,B_est))
    
    # Return dictionary with all fitted parameters and fitting range
    result = {
        'tau': tau_est,
        'A': A_est,
        'B': B_est,
        'tau_err': tau_err,
        'A_err': A_err,
        'B_err': B_err,
        'fit_start_ms': fit_start_ms,
        'fit_time_range': (t_fit[0], t_fit[-1]),
        'n_fit_points': len(t_fit)
    }
    return result



#...!...!.................... 
def compute_mean_crosscov_fastV2(spikes, max_lag_ms=None):
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
    '''
    #dt  = dt_ms/1000.0
    #if max_lag_ms is None:
    max_lag = N-1
    #else:
    #        max_lag = min(int(max_lag_ms/dt_ms), N-1)
    '''
    #dt = dt_ms/1000.0
    #if max_lag_ms is None:
    #    max_lag = N-1
    #else:
    max_lag = min(max_lag_ms, N-1)
        
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
    #times = np.arange(L,dtype=np.float32) #* dt
    return L, C



#...!...!.................... 

def compute_mean_autocovV2(spikes, dt_ms, max_lag_ms=None):
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
    #dt = dt_ms/1000.0
    #if max_lag_ms is None:
    #    max_lag = N-1
    #else:
    max_lag = min(int(max_lag_ms/dt_ms), N-1)
    Csum = np.zeros(max_lag+1, dtype=float)

    L=max_lag+1
    for i in range(nFeat):
        s = spikes[i].astype(float)
        μ = s.mean()
        s0 = s - μ
        # unbiased autocov for lags 0..max_lag
        for k in range(L):
            Csum[k] += np.dot(s0[:N-k], s0[k:])/(N-k)

    C = Csum / nFeat
    #times = np.arange(max_lag+1) * dt
    return L, C



# example
#=================================
#  M A I N 
#=================================

if __name__ == "__main__":
    C = [
        [1, 9, 5, 0],
        [0, 2, 0, 6],
        [7, 0, 3, 0],
        [0, 8, 0, 4]
    ]
    print('\ndaleMatrix_index_partition C:',C)
    C = np.array(C)
    Ldia, Lexc, Lzexc, Linh, Lzinh = daleMatrix_index_partition(C)

    print("Ldia indices   :", Ldia)
    print("diag vals      :", C[Ldia])
    print("E nonzeros idx :", Lexc)
    print("E nonzeros vals:", C[Lexc])
    print("E zeros idx    :", Lzexc)
    print("E zeros vals   :", C[Lzexc])
    print("I nonzeros idx :", Linh)
    print("I nonzeros vals:", C[Linh])
    print("I zeros idx    :", Lzinh)
    print("I zeros vals   :", C[Lzinh])

    #= = = = = = = = = = = = = = = = = 
    # build a toy (n,2) array
    A = np.array([[1,2],
                  [3,4],
                  [5,6],
                  [7,8]], dtype=float)

    stats, Xp, Yp = residual_stats(A)
    print('\nresidual_stats:')
    print("stats     :", stats)
    print("X′        :", Xp)
    print("Y′        :", Yp)
    print("means of X′,Y′:", Xp.mean(), Yp.mean())

   





