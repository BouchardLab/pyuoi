#!/usr/bin/env python3
"""
This script loads LASSO connectivity matrices from K bootstraps of real and shuffled data.
Diagonal elements are separated to compute mean and std across real-data bootstraps.
For off-diagonal edges, median magnitudes from real bootstraps are compared to a pooled null distribution
from shuffled bootstraps within each row to compute empirical p-values.
Row-wise Benjamini–Hochberg FDR is applied to select statistically significant edges at the given alpha level.
FDR = False Discovery Rate - a statistical method for controlling errors when testing many hypotheses simultaneously.
"""
import argparse
import os
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import pprint
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from PlotterFitEval import Plotter



def compare_triplets(Rec, trueD):
    """
    Compares two triplet arrays and returns True Positives, False Positives, and False Negatives.
    
    Parameters:
    Rec : array of shape (n, 3) with [i, j, value] triplets (Reconstructed/Predicted)
    trueD : array of shape (m, 3) with [i, j, value] triplets (Measured/Ground Truth)
    
    Returns:
    TP : array of shape (k, 4) with [i, j, rec_value, meas_value] for matching indices
    FP : array of shape (p, 3) with [i, j, value] for indices only in Rec
    FN : array of shape (q, 3) with [i, j, value] for indices only in Meas
    """
    # Convert to dictionaries for efficient lookup
    rec_dict = {(int(row[0]), int(row[1])): row[2] for row in Rec}
    true_dict = {(int(row[0]), int(row[1])): row[2] for row in trueD}
    
    # Get sets of indices
    rec_indices = set(rec_dict.keys())
    true_indices = set(true_dict.keys())
    
    # Find TP, FP, FN indices
    tp_indices = rec_indices & true_indices  # Intersection
    fp_indices = rec_indices - true_indices  # In Rec but not in Meas
    fn_indices = true_indices - rec_indices  # In Meas but not in Rec
    
    # Build output arrays
    TP = np.array([[i, j, rec_dict[(i,j)], true_dict[(i,j)]] 
                   for i, j in sorted(tp_indices)])
    FP = np.array([[i, j, rec_dict[(i,j)]] 
                   for i, j in sorted(fp_indices)])
    FN = np.array([[i, j, true_dict[(i,j)]] 
                   for i, j in sorted(fn_indices)])
    
    # Handle empty arrays
    if len(TP) == 0:
        TP = np.empty((0, 4))  # i,j,vr,vt
    if len(FP) == 0:
        FP = np.empty((0, 3))  # i,j,vr
    if len(FN) == 0:
        FN = np.empty((0, 3))   # i,j,vr,vt

    print('comp tripl  TP=%d, FP=%d, FN=%d'%(TP.shape[0],FP.shape[0],FN.shape[0]))
    #print('shapes tripl  TP=%s, FP=%s, FN=%s'%(TP.shape,FP.shape,FN.shape))
    return [TP, FP, FN]

def eval_tagged_edges_4_simu(fitD, trueD):
    #print(sorted(trueD))
    At=trueD['A_true']
    Bt=trueD['B_true']
    EposT=get_offdiag_triplets(At,True)
    EnegT=get_offdiag_triplets(At,False)
    print('True num edges  pos=%d  neg=%d'%(EposT.shape[0],EnegT.shape[0]))

    Ar=fitD['A_avr']  # reco
    Br=fitD['B_avr']  # reco
    EposR=get_offdiag_triplets(Ar,True)
    EnegR=get_offdiag_triplets(Ar,False)
    print('Reco num edges  pos=%d  neg=%d'%(EposR.shape[0],EnegR.shape[0]))

    #... zip diagonal
    diag_At = np.diag(At)
    diag_Ar = np.diag(Ar)
   
    evalD={}
    evalD['pos']=compare_triplets(EposR, EposT)
    evalD['neg']=compare_triplets(EnegR, EnegT)
    evalD['diag']=np.column_stack([diag_Ar, diag_At])
    evalD['bterm']=np.column_stack([Br, Bt])

    return  evalD
    
def get_offdiag_triplets(A, isPos=True):
    """
    Returns positive/negative, off-diagonal elements of 2D array A as array of [i, j, value] triplets.
    
    Parameters:
    A : 2D numpy array
    isPos : bool, if True only return positive values (>0), if False return all negative values
    
    Returns:
    Array of shape (n, 3) where n is the number of valid off-diagonal elements
    Each row is [row_index, col_index, value]
    """
    # Create mask for off-diagonal elements
    offdiag_mask = ~np.eye(A.shape[0], A.shape[1], dtype=bool)
    
    # Add value condition based on isPos
    if isPos:
        value_mask = A > 0
    else:
        value_mask = A < 0
    
    # Combine masks
    mask = value_mask & offdiag_mask
    
    i_indices, j_indices = np.where(mask)
    values = A[i_indices, j_indices]
    
    return np.column_stack([i_indices, j_indices, values])
    
def edge_selector_fdr(A_edges_real_list, A_edges_shuf_list, alpha=0.01):
    """
    Apply row-wise FDR to select significant edges.

    Parameters
    ----------
    A_edges_real_list : list of np.ndarray
        List of K real-data edge matrices (off-diagonals only).
    A_edges_shuf_list : list of np.ndarray
        List of K shuffled-data edge matrices (off-diagonals only).
    alpha : float
        FDR control level.

    Returns
    -------
    W_edges : np.ndarray
        Binary mask of significant edges (same shape as A_edges_real).
    W_pval : np.ndarray
        P-value matrix for each edge.
    summary : dict
        Summary statistics.
    """

    K = len(A_edges_real_list)
    assert K == len(A_edges_shuf_list), "Mismatch in number of real and shuffled bootstraps"

    # Stack into arrays: shape (K, N, N)
    A_real = np.stack(A_edges_real_list, axis=0)
    A_shuf = np.stack(A_edges_shuf_list, axis=0)

    N = A_real.shape[1]

    # Statistic: median magnitude across bootstraps for each edge
    stat_obs = np.median(np.abs(A_real), axis=0)  # shape (N, N)

    # Null distribution: pool shuffled bootstraps per row
    W_pval = np.ones((N, N))
    W_edge_mask = np.zeros((N, N), dtype=bool)
    #W_edge_mask is a binary mask of significant edges after row‑wise FDR.

    # Statistics tracking    
    edges_per_row = []

    for i in range(N):
        # Pool null values for row i from shuffled data
        null_vals_row = np.abs(A_shuf[:, i, :]).ravel()        
        # Compute p-values for all j in this row
        for j in range(N):
            if i == j:
                continue
            obs_val = stat_obs[i, j]
            # Empirical p-value
            pval = np.mean(null_vals_row >= obs_val)
            W_pval[i, j] = pval

        # Apply FDR for this row
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        #......  per‑row FDR selection, not global FDR
        reject, _ = fdrcorrection(W_pval[i, mask], alpha=alpha)
        W_edge_mask[i, mask] = reject
        
        # Count selected edges for this row
        edges_per_row.append(int(reject.sum()))

    # Additional statistics
    edges_per_row = np.array(edges_per_row)
    
    # Count zeros in real data matrices
    total_elements = A_real.size
    zero_elements = np.sum(A_real == 0.0)
    sparsity_A_real = zero_elements / total_elements
    
    # Compute sparsity of W_edge_mask (fraction of zeros in off-diagonal elements)
    off_diag_elements = N * (N - 1)  # Total off-diagonal elements
    selected_edges = int(W_edge_mask.sum()) - N  # Subtract diagonal (always True)
    sparsity_W_mask = 1.0 - (selected_edges / off_diag_elements)
    
    # Compute pooled values from A_shuf dimensions: A_shuf shape is (K, N, N)
    # For each row i, we pool A_shuf[:, i, :].ravel() which gives K*N values
    pooled_values_per_row = K * N
    pooled_values_total = N * pooled_values_per_row  # N rows total

    summary = {
        "alpha": alpha,
        "num_bootstraps": K,
        "matrix_size": N,
        "num_significant_edges": int(W_edge_mask.sum()) - N,  # Exclude diagonal
        "pooled_values_per_row": int(pooled_values_per_row),
        "pooled_values_total": int(pooled_values_total),
        "zero_elements_A_lasso": int(zero_elements),
        "total_elements_A_lasso": int(total_elements),
        "sparsity_A_lasso": float(sparsity_A_real),
        "sparsity_W_edge_mask": float(sparsity_W_mask),
        "edges_per_row_avg": float(edges_per_row.mean()),
        "edges_per_row_std": float(edges_per_row.std()),
        "edges_per_row_min": int(edges_per_row.min()),
        "edges_per_row_max": int(edges_per_row.max()),
    }

    return W_edge_mask, W_pval, summary


def load_bootstrap_data(dataName, dataPath, K, verb=1):
    """
    Load all bootstrap data from real and shuffled files.
    
    Parameters
    ----------
    dataName : str
        Base name for the dataset
    dataPath : str
        Path to the data directory
    K : int
        Number of bootstraps
    verb : int
        Verbosity level
        
    Returns
    -------
    A_edges_real_list : list
        List of real edge matrices (off-diagonal only)
    A_edges_shuf_list : list
        List of shuffled edge matrices (off-diagonal only)
    A_real_list : list
        List of full real A matrices
    B_real_list : list
        List of real B vectors
    output_meta : dict
        Metadata from first bootstrap file
    """
    
    A_edges_real_list = []
    A_edges_shuf_list = []
    A_real_list = []
    B_real_list = []

    for k in range(K):
        # Real data
        real_file = os.path.join(dataPath, f"{dataName}-boot{k}.lassoFit.npz")
        data_real, meta_real = read_data_npz(real_file, verb=(k==0 and verb>=2))
        A_real = data_real['A_lasso']
        B_real = data_real['B_lasso']
        
        # Store full matrices for averaging
        A_real_list.append(A_real)
        B_real_list.append(B_real)
        
        # Store edges for FDR analysis
        edges_real = A_real.copy()
        np.fill_diagonal(edges_real, 0.0)
        A_edges_real_list.append(edges_real)

        # Store metadata from first bootstrap for output
        if k == 0:
            output_meta = meta_real
            output_big1=data_real

        # Shuffled data
        shuf_file = real_file.replace('boot','shuf')
        data_shuf, _ = read_data_npz(shuf_file, verb=(k==0 and verb>=2))
        A_shuf = data_shuf['A_lasso']
        edges_shuf = A_shuf.copy()
        np.fill_diagonal(edges_shuf, 0.0)

        A_edges_shuf_list.append(edges_shuf)
    
    if verb >= 1:
        print(f"Loaded {K} bootstrap files with real and shuffled data")
        
    return A_edges_real_list, A_edges_shuf_list, A_real_list, B_real_list, output_meta,output_big1


def load_auxiliary_plotting_data(fitMD, maskMD, dataName, dataPath, alpha):
    """
    Load auxiliary data needed for plotting (spike data, ground truth, metadata).
    
    Parameters
    ----------
    fitMD : dict
        Fit metadata dictionary
    maskMD : dict
        Mask metadata dictionary  
    dataName : str
        Dataset name for short_name
    dataPath : str
        Path to data directory
    alpha : float
        FDR alpha value for metadata
        
    Returns
    -------
    spikeD : dict
        Spike data dictionary
    trueD : dict or None
        Ground truth data (None if not simDale)
    MD : dict
        Combined metadata for plotting
    """
    
    # Load spike data for frequency sorting
    spikeF = fitMD['fit_lasso']['lassoFit_input_name']    
    spikesFF = os.path.join(dataPath, f"{spikeF}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=0)
    
    # Load ground truth data if simulated Dale data
    trueD = None
    if fitMD['data_type']=='simDale':
        truthFF = os.path.join(dataPath, f"{spikeF}.simTruth.npz")
        trueD, trueMD = read_data_npz(truthFF, verb=0)    
        # Combine metadata just for plotter
        MD = {**fitMD, **trueMD, 'short_name': dataName}
    else:
        MD = {**fitMD, 'short_name': dataName}
    
    # Add mask metadata and FDR method info
    MD.update(maskMD)
    MD['edge_selection_method'] = 'fdr'
    MD['fdr_alpha'] = alpha
    
    return spikeD, trueD, MD


#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Row-wise FDR edge selection from LASSO bootstraps")
    parser.add_argument("--dataName", type=str, required=True, help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/")
    parser.add_argument("--verb", "-v", type=int, default=1, help="Verbosity level")
    parser.add_argument("--num_bootstraps", type=int, required=True, help="Number of bootstraps (K)")
    parser.add_argument("--alpha", type=float, default=0.01, help="FDR significance level")
    
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="f", help="Plot types to show: a=structure, e=experiment_eigen, f=freqSortA_histos")
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    args = parser.parse_args()
    print(vars(args))
        
    if args.outPath is None:   
        args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
 
    dataName = args.dataName
    dataPath = args.dataPath
    K = args.num_bootstraps
    alpha = args.alpha

    # Load all bootstrap data
    A_edges_real_list, A_edges_shuf_list, A_real_list, B_real_list, output_meta,output_big1 = load_bootstrap_data(
        dataName, dataPath, K, args.verb
    )

    # Apply row-wise FDR to edges
    W_mask, W_pval, summary = edge_selector_fdr(A_edges_real_list, A_edges_shuf_list, alpha=alpha)

    print("FDR selection summary:"); pprint.pprint(summary)
    
    # Compute averages and standard deviations from real data bootstraps
    A_stack = np.stack(A_real_list, axis=0)  # shape (K, N, N)
    B_stack = np.stack(B_real_list, axis=0)  # shape (K, N)
    
    A_avr = np.mean(A_stack, axis=0)
    A_std = np.std(A_stack, axis=0)
    B_avr = np.mean(B_stack, axis=0)
    B_std = np.std(B_stack, axis=0)
    
    # Create full W_mask that applies only to off-diagonal elements
    N = A_avr.shape[0]
    W_mask_full = np.eye(N, dtype=bool)  # Start with diagonal = True (keep diagonal)
    # Apply FDR mask to off-diagonal elements only
    off_diag_mask = ~np.eye(N, dtype=bool)
    W_mask_full[off_diag_mask] = W_mask[off_diag_mask]

    A_avr[~W_mask_full]=0.  # now none-existing edges are 0
    
    print(f"Computed averages and std from {K} real bootstraps")
    print(f"A_avr shape: {A_avr.shape}, B_avr shape: {B_avr.shape}")
    print(f"Mask preserves diagonal and selects {W_mask.sum()} significant off-diagonal edges")
    
    # Prepare output data
    output_data = {
        'W_mask': W_mask_full,
        'A_avr': A_avr,
        'A_std': A_std,
        'B_avr': B_avr,
        'B_std': B_std,
        'W_pval': W_pval,
        'summary': summary
    }
    for xx in [ 'losses_total', 'losses_epochs', 'losses_wo_L1']:
        output_data[xx]=output_big1[xx]

    output_meta['edge_selector']={'selector_type':'FDR', 'alpha':args.alpha}
 
    # Save results
    output_file = os.path.join(dataPath, f"{dataName}-selFdr.lassoFit.npz")
    write_data_npz(output_data, output_file, metaD=output_meta)
    print(f"FDR results saved to: {output_file}")
    
    # Generate plots if requested
    if args.showPlots:
        print(f"\nGenerating plots: {args.showPlots}")
        
        # Prepare plotting data (compatible with eval_fitLasso.py structure)
        fitD = output_data.copy()  # Use our processed output data as fitD
        fitMD = output_meta
        
        # Rename records so select_edges_from_fitLasso() has the expected names
        fitD['A_lasso'] = A_avr.copy()
        fitD['B_lasso'] = B_avr.copy()
        
        # Load auxiliary data needed for plotting
        maskMD={}
        spikeD, trueD, MD = load_auxiliary_plotting_data(fitMD, maskMD, dataName, dataPath, alpha)

        if fitMD['data_type']=='simDale':
            evalD=eval_tagged_edges_4_simu(fitD,trueD)            
        
        # adjustment for plotting

        fitD['single_rates']=spikeD['single_rates']
        
        # Setup plotter
        args.prjName = dataName 
        plot = Plotter(args)
        
        # Generate plots based on showPlots argument
        if 'a' in args.showPlots:
            plot.summary_fitLasso(fitD,MD,figId=1)

        if 'b' in args.showPlots:
            assert  fitMD['data_type']=='simDale'
            plot.residuals(evalD,MD,figId=2)

        if 'c' in args.showPlots:
            plot.freqSortA_histos(fitD, MD, spikeD, figId=3)
      
        if 'd' in args.showPlots:            
            plot.experiment_eigen(fitD, MD, figId=4)
        
        plot.display_all()
        print("Plotting completed.")

if __name__ == "__main__":
    main()
