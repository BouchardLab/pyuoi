
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
