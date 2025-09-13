#!/usr/bin/env python3
"""
select_edges.py - Bootstrap edge selection for Lasso Poisson results

This script processes multiple bootstrap results to identify statistically 
significant edges in the learned connectivity matrix.

FDR = False Discovery Rate - a statistical method for controlling errors when testing many hypotheses simultaneously.

Original method: nSig was applied to signal-to-noise ratio of individual edges
FDR method: alpha controls family-wise error rate across all edges simultaneously
So FDR with α=0.05 is actually more stringent than the original method with nSig=2.0 because it accounts for testing thousands of edges at once!
Recommended starting points:
Exploratory: --alpha 0.05 (standard significance level)
Conservative: --alpha 0.01 (1% false positive rate)
Very conservative: --alpha 0.001 (0.1% false positive rate)

"""

import os
import argparse
import numpy as np
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from UtilEdgeSelector import edge_selector_s2n, edge_selector_fdr, edge_selector_fdr_effect_size
from PlotterFitEval import Plotter
from UtilDalePoisson import select_edges_from_fitLasso

def main():
    parser = argparse.ArgumentParser(description="Bootstrap edge selection for Lasso Poisson results")
    parser.add_argument("--dataName", type=str, required=True, help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="Path to the data directory")
    parser.add_argument("-v", "--verb", type=int, default=1, help="Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)")
    parser.add_argument("--num_bootstraps", type=int, required=True, help="Number of bootstrap files to process")
    parser.add_argument("--method", type=str, default="s2n", choices=["s2n", "fdr1", "fdr2"], help="Edge selection method: s2n (signal-to-noise), fdr1 (basic FDR), fdr2 (FDR + effect size)")
    parser.add_argument("--nSig", type=float, default=2.0, help="Significance threshold for s2n method (number of standard deviations)")
    parser.add_argument("--minW", type=float, default=0.07, help="Minimum weight threshold for s2n method")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for FDR methods (e.g., 0.05)")
    parser.add_argument("--effect_size_factor", type=float, default=2.0, help="Effect size factor for fdr2 method (median + factor*MAD). Higher = more restrictive ")
    parser.add_argument('-A',"--ampl_thres", type=float, default=[0.10],nargs='+', help=" inh< tht0, exct>th1 of accepted off-diagonal edge")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="ab", help="Plot types to show: a=structure, b=distributions, c=reconstruction, d=category, d=A-matrix histograms")
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    
    args = parser.parse_args()
    print(vars(args))
    # Validate method-specific parameters
    if args.method == "s2n" and (args.nSig <= 0 or args.minW <= 0):
        print("Error: s2n method requires positive nSig and minW values")
        return 1
    elif args.method in ["fdr1", "fdr2"] and args.alpha <= 0:
        print("Error: FDR methods require alpha > 0")
        return 1
    
    # Process plotting arguments
    if len(args.ampl_thres)==1:
        args.ampl_thres=[-args.ampl_thres[0],args.ampl_thres[0]]
    
    if args.outPath is None:   
        args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
    
    if args.verb >= 1:
        print("=" * 60)
        print("BOOTSTRAP EDGE SELECTION")
        print("=" * 60)
        print(f"Data: {args.dataName}")
        print(f"Bootstraps: {args.num_bootstraps}")
        print(f"Method: {args.method}")
        print()
    
    # Collect data from all bootstrap files
    B_lasso_list = []
    A_lasso_list = []
    
    if args.verb >= 2:
        print("Loading bootstrap files:")
    
    for k in range(1, args.num_bootstraps + 1):
        filename = f"{args.dataName}-boot{k}.lassoFit.npz"
        filepath = os.path.join(args.dataPath, filename)
        
        if args.verb >= 2:
            print(f"  Loading {filename}")
        
        # Load data
        data, metadata = read_data_npz(filepath, verb=0)
        
        # Extract arrays
        B_lasso_list.append(data['B_lasso'])
        A_lasso_list.append(data['A_lasso'])
        
        # Store metadata from first file
        if k == 1:
            first_metadata = metadata
    
    if args.verb >= 1:
        print(f"Successfully loaded {len(B_lasso_list)} bootstrap files")
        print()
    
    # Process B_lasso: compute element-wise mean and std
    if args.verb >= 2:
        print("Processing B_lasso arrays...")
    
    B_lasso_stack = np.stack(B_lasso_list, axis=0)  # Shape: (K, M)
    B_lasso_mean = np.mean(B_lasso_stack, axis=0)
    B_lasso_std = np.std(B_lasso_stack, axis=0, ddof=1)
    
    if args.verb >= 2:
        print(f"B_lasso shape: {B_lasso_mean.shape}")
        print(f"B_lasso mean range: [{B_lasso_mean.min():.4f}, {B_lasso_mean.max():.4f}]")
        print()
    
    # Process A_lasso: split into diagonal and off-diagonal
    if args.verb >= 2:
        print("Processing A_lasso arrays...")
    
    A_shape = A_lasso_list[0].shape
    M = A_shape[0]
    
    # Extract diagonal elements from all bootstraps
    A_diag_list = []
    A_edges_list = []
    
    for A_lasso in A_lasso_list:
        # Extract diagonal
        A_diag = np.diag(A_lasso)
        A_diag_list.append(A_diag)
        
        # Create off-diagonal matrix (set diagonal to 0)
        A_edges = A_lasso.copy()
        np.fill_diagonal(A_edges, 0)
        A_edges_list.append(A_edges)
    
    # Process diagonal elements
    A_diag_stack = np.stack(A_diag_list, axis=0)  # Shape: (K, M)
    A_diag_mean = np.mean(A_diag_stack, axis=0)
    A_diag_std = np.std(A_diag_stack, axis=0, ddof=1)
    
    if args.verb >= 2:
        print(f"A_diag shape: {A_diag_mean.shape}")
        print(f"A_diag mean range: [{A_diag_mean.min():.4f}, {A_diag_mean.max():.4f}]")
    
    # Process off-diagonal elements with edge selection
    if args.verb >= 2:
        print(f"Applying edge selection to A_edges (shape: {A_shape})...")
        print()
    
    # Choose edge selection method
    if args.method == "s2n":
        W_edges_mean, W_edges_std, edge_stats = edge_selector_s2n(
            A_edges_list, args.nSig, args.minW, args.verb
        )
        method_name = "s2n"
    elif args.method == "fdr1":
        W_edges_mean, W_edges_std, edge_stats = edge_selector_fdr(
            A_edges_list, args.alpha, args.verb
        )
        method_name = "fdr1"
    elif args.method == "fdr2":
        W_edges_mean, W_edges_std, edge_stats = edge_selector_fdr_effect_size(
            A_edges_list, args.alpha, args.effect_size_factor, args.verb
        )
        method_name = "fdr2"
    else:
        print(f"Error: Unknown method '{args.method}'")
        return 1
    
    num_edges = edge_stats['acc_positive'] + edge_stats['acc_negative']
    
    # Combine diagonal and off-diagonal results
    # Insert diagonal values back into the edge matrices
    np.fill_diagonal(W_edges_mean, A_diag_mean)
    np.fill_diagonal(W_edges_std, A_diag_std)
    
    # Prepare output data
    output_data = {
        'B_lasso_mean': B_lasso_mean,
        'B_lasso_std': B_lasso_std,
        'W_edges_mean': W_edges_mean,
        'W_edges_std': W_edges_std
    }
    
    # Update metadata with selection parameters
    output_metadata = first_metadata
    edge_selection_meta = {
        'num_bootstraps': args.num_bootstraps,
        'method': method_name,
        'num_selected_edges': int(num_edges),
        'total_off_diagonal': int(M * M - M),
        'selection_fraction': float(num_edges) / (M * M - M)
    }
    
    # Add method-specific parameters
    if args.method == "s2n":
        edge_selection_meta['nSig'] = args.nSig
        edge_selection_meta['minW'] = args.minW
    elif args.method == "fdr1":
        edge_selection_meta['alpha'] = args.alpha
    elif args.method == "fdr2":
        edge_selection_meta['alpha'] = args.alpha
        edge_selection_meta['effect_size_factor'] = args.effect_size_factor
        
    output_metadata['fit_lasso']['edge_selection'] = edge_selection_meta
    
    # Add detailed edge selection statistics (method-dependent)
    edg_select_stats = {
        'offdiag': int(edge_stats['offdiag']),
        'acc_positive': int(edge_stats['acc_positive']),
        'acc_negative': int(edge_stats['acc_negative'])
    }
    
    if args.method == "s2n":
        # S2N method counters
        edg_select_stats.update({
            'small_xmean': int(edge_stats['small_xmean']),
            'small_nsig': int(edge_stats['small_nsig'])
        })
    elif args.method == "fdr1":
        # FDR1 method counters
        edg_select_stats.update({
            'tested': int(edge_stats['tested']),
            'low_variance': int(edge_stats['low_variance']),
            'significant': int(edge_stats['significant']),
            'neurons_tested': int(edge_stats['neurons_tested']),
            'neurons_with_edges': int(edge_stats['neurons_with_edges'])
        })
    elif args.method == "fdr2":
        # FDR2 method counters (includes effect size filtering)
        edg_select_stats.update({
            'tested': int(edge_stats['tested']),
            'low_variance': int(edge_stats['low_variance']),
            'significant_fdr': int(edge_stats['significant_fdr']),
            'effect_filtered': int(edge_stats['effect_filtered']),
            'significant': int(edge_stats['significant']),
            'neurons_tested': int(edge_stats['neurons_tested']),
            'neurons_with_edges': int(edge_stats['neurons_with_edges'])
        })
        
    output_metadata['fit_lasso']['edg_select'] = edg_select_stats
    
    # Save results
    output_filename = f"{args.dataName}-select{args.num_bootstraps}.lassoFit.npz"
    output_filepath = os.path.join(args.dataPath, output_filename)
    
    if args.verb >= 1:
        print(f"Saving results to: {output_filename}")
        print(f"Selected {num_edges} significant edges out of {M*M-M} possible")
        print()
    
    
    if args.verb >= 1:
        print("=" * 60)
        print("BOOTSTRAP EDGE SELECTION COMPLETED")
        print("=" * 60)
        print(f"Output file: {output_filename}")
        print(f"B_lasso: {B_lasso_mean.shape} (mean and std)")
        print(f"A_matrix: {W_edges_mean.shape} (selected edges + diagonal)")
        print(f"Total significant edges: {num_edges}/{M*M-M} ({100*num_edges/(M*M-M):.1f}%)")
        print()
        
    write_data_npz(output_data, output_filepath, metaD=output_metadata)
    
    # Prepare plotting data (always executed)
    # Create mask data using same structure as eval_fitLasso.py
    fitD = output_data  # Use our processed output data as fitD
    fitMD = output_metadata
    
    # Rename records so select_edges_from_fitLasso() has the expected names
    fitD['A_lasso'] = fitD['W_edges_mean'].copy()
    fitD['B_lasso'] = fitD['B_lasso_mean']
    
    # Set exactly 0 values to NaN so they are not displayed in plots
    fitD['A_lasso'][fitD['A_lasso'] == 0.0] = np.nan
    
    maskD, maskMD = select_edges_from_fitLasso(fitD, args.ampl_thres)
    maskMD['fit_lasso'] = fitMD['fit_lasso']
    
    # Load spike data for frequency sorting (get spike file name from first bootstrap)
    spikeF = fitMD['fit_lasso']['lassoFit_input_name']    
    spikesFF = os.path.join(args.dataPath, f"{spikeF}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF)
    
    if fitMD['data_type']=='simDale':
        # Load ground truth data
        truthFF = os.path.join(args.dataPath, f"{spikeF}.simTruth.npz")
        trueD, trueMD = read_data_npz(truthFF)    
        # Combine metadata just for plotter
        MD = {**fitMD, **trueMD, 'short_name': args.dataName}
    else:
        MD = {**fitMD,  'short_name': args.dataName}
    
    MD.update(maskMD)
    
    # Add edge selection method to metadata for plotting
    MD['edge_selection_method'] = method_name
    
    # Setup plotter
    args.prjName = args.dataName 
    plot = Plotter(args)
    
    # Generate plots based on showPlots argument
    if 'a' in args.showPlots:
        # Plot correlation after threshold (requires simDale data type)
        assert fitMD['data_type']=='simDale'
        plot.correl_after_thresh(trueD,fitD,maskD,MD,figId=1)
    
    if 'e' in args.showPlots:
        plot.experiment_eigen(fitD,MD, figId=5)
    
    if 'f' in args.showPlots:
        plot.freqSortA_histos(fitD, MD, spikeD, figId=2)
    
    plot.display_all()
  
    return 0

if __name__ == "__main__":
    exit(main())
