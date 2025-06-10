#!/usr/bin/env python3

import numpy as np

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
import numpy as np

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




#...!...!.................... 



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
