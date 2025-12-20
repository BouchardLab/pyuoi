"""
Edge selection utilities for bootstrap analysis.

This module provides different statistical methods for selecting
significant edges from bootstrap connectivity matrix samples.
"""

import numpy as np

def edge_selector_s2n(A_edges_list, nSig, minW, verb=0):
    """
    Select statistically significant edges from bootstrap samples using signal-to-noise ratio.
    
    Args:
        A_edges_list: List of K A_edges arrays (off-diagonal elements)
        nSig: Significance threshold (number of standard deviations)
        minW: Minimum weight threshold
        verb: Verbosity level
        
    Returns:
        W_edges_mean: Mean values of significant edges (others set to 0)
        W_edges_std: Std values of significant edges (others set to 0) 
        selection_stats: Dictionary with detailed selection statistics
    """
    if len(A_edges_list) == 0:
        raise ValueError("Empty A_edges_list provided")
    
    # Get shape from first array
    shape = A_edges_list[0].shape
    K = len(A_edges_list)
    
    # Initialize output arrays
    W_edges_mean = np.zeros(shape)
    W_edges_std = np.zeros(shape)
    
    # Initialize counters
    stats = {
        'offdiag': 0,      # Total off-diagonal elements processed
        'small_xmean': 0,  # Rejected by first filter (abs(xmean) + nSig*xstd < minW)
        'small_nsig': 0,   # Rejected by second filter (abs(xmean)/xstd < nSig)
        'acc_positive': 0, # Accepted edges with positive mean
        'acc_negative': 0  # Accepted edges with negative mean
    }
    
    if verb >= 1:
        print("=" * 60)
        print("EDGE SELECTION: Signal-to-Noise Ratio Method")
        print("=" * 60)
        print(f"Parameters: nSig={nSig}, minW={minW}")
        print(f"Criteria: |mean| + {nSig}*std >= {minW} AND |mean|/std >= {nSig}")
        print(f"Processing {shape} edge positions across {K} bootstraps")
        print()
    
    # Loop over all edge positions
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Skip diagonal elements (should already be removed, but safety check)
            if i == j:
                continue
            
            # Count off-diagonal elements
            stats['offdiag'] += 1
                
            # Collect values from all K bootstrap samples
            xV = np.array([A_edges[i, j] for A_edges in A_edges_list])
            
            # Compute statistics
            xmean = np.mean(xV)
            xstd = np.std(xV, ddof=1)  # Use sample std (N-1 denominator)
            
            # Apply significance filters
            abs_xmean = abs(xmean)
            
            # Filter 1: Mean + nSig*std must exceed minimum weight
            if abs_xmean + nSig * xstd < minW:
                stats['small_xmean'] += 1
                continue
                
            # Filter 2: Signal-to-noise ratio must exceed nSig
            if xstd > 0 and abs_xmean / xstd < nSig:
                stats['small_nsig'] += 1
                continue
                
            # Edge passed both filters - store it
            W_edges_mean[i, j] = xmean
            W_edges_std[i, j] = xstd
            
            # Count accepted edges by sign
            if xmean >= 0:
                stats['acc_positive'] += 1
            else:
                stats['acc_negative'] += 1
            
            if verb >= 3:
                print(f"Significant edge ({i},{j}): mean={xmean:.4f}, std={xstd:.4f}, SNR={abs_xmean/xstd:.2f}")
    
    # Calculate total accepted
    num_edges = stats['acc_positive'] + stats['acc_negative']
    
    if verb >= 1:
        print("RESULTS:")
        print(f"  Total off-diagonal elements: {stats['offdiag']}")
        print(f"  Rejected (small magnitude):  {stats['small_xmean']} ({100*stats['small_xmean']/stats['offdiag']:.1f}%)")
        print(f"  Rejected (low significance): {stats['small_nsig']} ({100*stats['small_nsig']/stats['offdiag']:.1f}%)")
        print(f"  Accepted (positive):         {stats['acc_positive']} ({100*stats['acc_positive']/stats['offdiag']:.1f}%)")
        print(f"  Accepted (negative):         {stats['acc_negative']} ({100*stats['acc_negative']/stats['offdiag']:.1f}%)")
        print(f"  TOTAL SELECTED:              {num_edges}/{stats['offdiag']} ({100*num_edges/stats['offdiag']:.1f}%)")
        
        # Verification
        total_processed = stats['small_xmean'] + stats['small_nsig'] + stats['acc_positive'] + stats['acc_negative']
        if total_processed != stats['offdiag']:
            print(f"  WARNING: Count mismatch! Expected {stats['offdiag']}, got {total_processed}")
        print()
    
    return W_edges_mean, W_edges_std, stats

def edge_selector_fdr(A_edges_list, alpha=0.05, verb=0):
    """
    Select statistically significant edges using row-wise FDR correction.
    
    Apply FDR correction independently for each neuron (each row of matrix A).
    This accounts for different statistical accuracy due to varying firing rates.
    
    Args:
        A_edges_list: List of K A_edges arrays (off-diagonal elements)
        alpha: Significance level for FDR correction
        verb: Verbosity level
        
    Returns:
        W_edges_mean: Mean values of significant edges (others set to 0)
        W_edges_std: Std values of significant edges (others set to 0) 
        selection_stats: Dictionary with detailed selection statistics
    """
    if len(A_edges_list) == 0:
        raise ValueError("Empty A_edges_list provided")
    
    try:
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        raise ImportError("FDR method requires scipy and statsmodels. Install with: pip install scipy statsmodels")
    
    # Get shape from first array
    shape = A_edges_list[0].shape
    K = len(A_edges_list)
    M = shape[0]  # Number of neurons
    
    # Initialize output arrays
    W_edges_mean = np.zeros(shape)
    W_edges_std = np.zeros(shape)
    
    # Initialize counters
    stats_dict = {
        'offdiag': 0,        # Total off-diagonal elements processed
        'tested': 0,         # Elements tested (had enough variance for t-test)
        'low_variance': 0,   # Skipped due to zero/low variance
        'significant': 0,    # Passed FDR correction
        'acc_positive': 0,   # Significant edges with positive mean
        'acc_negative': 0,   # Significant edges with negative mean
        'neurons_tested': 0, # Number of neurons that had testable edges
        'neurons_with_edges': 0  # Number of neurons that got significant edges
    }
    
    if verb >= 1:
        print("=" * 60)
        print("EDGE SELECTION: Row-wise FDR Multiple Testing Correction")
        print("=" * 60)
        print(f"Parameters: alpha={alpha}")
        print(f"Method: Benjamini-Hochberg FDR applied independently for each neuron")
        print(f"Processing {shape} edge positions across {K} bootstraps")
        print(f"Applying FDR independently for each of {M} neurons")
        print()
    
    # Process each neuron (row) independently
    for neuron_i in range(M):
        if verb >= 3:
            print(f"Processing neuron {neuron_i+1}/{M}")
        
        # Collect data for this neuron's incoming connections
        neuron_positions = []
        neuron_means = []
        neuron_stds = []
        neuron_p_values = []
        
        # Collect all incoming edges for this neuron (row neuron_i)
        for j in range(M):
            if neuron_i == j:  # Skip diagonal
                continue
                
            stats_dict['offdiag'] += 1
            
            # Collect values from all K bootstrap samples
            xV = np.array([A_edges[neuron_i, j] for A_edges in A_edges_list])
            
            # Compute statistics
            xmean = np.mean(xV)
            xstd = np.std(xV, ddof=1)
            
            neuron_positions.append((neuron_i, j))
            neuron_means.append(xmean)
            neuron_stds.append(xstd)
            
            # Perform one-sample t-test against null hypothesis (mean = 0)
            if xstd > 1e-10 and K > 1:
                t_stat, p_val = stats.ttest_1samp(xV, 0)
                neuron_p_values.append(p_val)
                stats_dict['tested'] += 1
            else:
                neuron_p_values.append(1.0)  # Non-significant p-value for low variance
                stats_dict['low_variance'] += 1
        
        # Apply FDR correction for this neuron's edges only
        if len(neuron_p_values) > 0:
            stats_dict['neurons_tested'] += 1
            
            # Count testable edges for this neuron
            testable_edges = sum(1 for p in neuron_p_values if p < 1.0)
            
            if testable_edges > 0:
                rejected, p_adj, alpha_sidak, alpha_bonf = multipletests(
                    neuron_p_values, alpha=alpha, method='fdr_bh'
                )
                
                neuron_significant_count = 0
                
                # Store significant edges for this neuron
                for k, (i, j) in enumerate(neuron_positions):
                    if rejected[k]:
                        W_edges_mean[i, j] = neuron_means[k]
                        W_edges_std[i, j] = neuron_stds[k]
                        stats_dict['significant'] += 1
                        neuron_significant_count += 1
                        
                        # Count by sign
                        if neuron_means[k] >= 0:
                            stats_dict['acc_positive'] += 1
                        else:
                            stats_dict['acc_negative'] += 1
                        
                        if verb >= 3:
                            print(f"  Significant edge ({i},{j}): mean={neuron_means[k]:.4f}, std={neuron_stds[k]:.4f}, p_adj={p_adj[k]:.4e}")
                
                if neuron_significant_count > 0:
                    stats_dict['neurons_with_edges'] += 1
                    
                if verb >= 3:
                    print(f"  Neuron {neuron_i}: {testable_edges} testable, {neuron_significant_count} significant")
    
    if verb >= 1:
        print("RESULTS:")
        print(f"  Total off-diagonal elements: {stats_dict['offdiag']}")
        print(f"  Testable (sufficient var):   {stats_dict['tested']} ({100*stats_dict['tested']/stats_dict['offdiag']:.1f}%)")
        print(f"  Low variance (skipped):      {stats_dict['low_variance']} ({100*stats_dict['low_variance']/stats_dict['offdiag']:.1f}%)")
        print(f"  Accepted (positive):         {stats_dict['acc_positive']} ({100*stats_dict['acc_positive']/stats_dict['offdiag']:.1f}%)")
        print(f"  Accepted (negative):         {stats_dict['acc_negative']} ({100*stats_dict['acc_negative']/stats_dict['offdiag']:.1f}%)")
        print(f"  TOTAL SELECTED:              {stats_dict['significant']}/{stats_dict['offdiag']} ({100*stats_dict['significant']/stats_dict['offdiag']:.1f}%)")
        print(f"  Neurons tested:              {stats_dict['neurons_tested']} out of {M}")
        print(f"  Neurons with edges:          {stats_dict['neurons_with_edges']} ({100*stats_dict['neurons_with_edges']/max(1,stats_dict['neurons_tested']):.1f}% of tested)")
        
        # Verification
        print(f"  Verification: {stats_dict['offdiag']} = {stats_dict['tested']} testable + {stats_dict['low_variance']} low-var")
        print()
    
    return W_edges_mean, W_edges_std, stats_dict

def edge_selector_fdr_effect_size(A_edges_list, alpha=0.05, effect_size_factor=1.5, verb=0):
    """
    Select statistically significant edges using row-wise FDR with automatic minimum effect size.
    
    Apply FDR correction independently for each neuron (each row of matrix A) combined
    with data-driven minimum effect size threshold computed per row.
    
    Args:
        A_edges_list: List of K A_edges arrays (off-diagonal elements)
        alpha: Significance level for FDR correction
        effect_size_factor: Factor for effect size threshold (median + factor*MAD). Higher = more restrictive.
        verb: Verbosity level
        
    Returns:
        W_edges_mean: Mean values of significant edges (others set to 0)
        W_edges_std: Std values of significant edges (others set to 0) 
        selection_stats: Dictionary with detailed selection statistics
    """
    if len(A_edges_list) == 0:
        raise ValueError("Empty A_edges_list provided")
    
    try:
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        raise ImportError("FDR method requires scipy and statsmodels. Install with: pip install scipy statsmodels")
    
    # Get shape from first array
    shape = A_edges_list[0].shape
    K = len(A_edges_list)
    M = shape[0]  # Number of neurons
    
    # Initialize output arrays
    W_edges_mean = np.zeros(shape)
    W_edges_std = np.zeros(shape)
    
    # Initialize counters
    stats_dict = {
        'offdiag': 0,           # Total off-diagonal elements processed
        'tested': 0,            # Elements tested (had enough variance for t-test)
        'low_variance': 0,      # Skipped due to zero/low variance
        'significant_fdr': 0,   # Passed FDR correction
        'effect_filtered': 0,   # Removed by effect size filter
        'significant': 0,       # Final count after both filters
        'acc_positive': 0,      # Significant edges with positive mean
        'acc_negative': 0,      # Significant edges with negative mean
        'neurons_tested': 0,    # Number of neurons that had testable edges
        'neurons_with_edges': 0 # Number of neurons that got significant edges
    }
    
    if verb >= 1:
        print("=" * 60)
        print("EDGE SELECTION: Row-wise FDR + Minimum Effect Size")
        print("=" * 60)
        print(f"Parameters: alpha={alpha}, effect_size_factor={effect_size_factor}")
        print(f"Method: Benjamini-Hochberg FDR + auto effect size per neuron")
        print(f"Processing {shape} edge positions across {K} bootstraps")
        print(f"Applying FDR + effect size independently for each of {M} neurons")
        print()
    
    # Process each neuron (row) independently
    for neuron_i in range(M):
        if verb >= 3:
            print(f"Processing neuron {neuron_i+1}/{M}")
        
        # Collect data for this neuron's incoming connections
        neuron_positions = []
        neuron_means = []
        neuron_stds = []
        neuron_p_values = []
        
        # Collect all incoming edges for this neuron (row neuron_i)
        for j in range(M):
            if neuron_i == j:  # Skip diagonal
                continue
                
            stats_dict['offdiag'] += 1
            
            # Collect values from all K bootstrap samples
            xV = np.array([A_edges[neuron_i, j] for A_edges in A_edges_list])
            
            # Compute statistics
            xmean = np.mean(xV)
            xstd = np.std(xV, ddof=1)
            
            neuron_positions.append((neuron_i, j))
            neuron_means.append(xmean)
            neuron_stds.append(xstd)
            
            # Perform one-sample t-test against null hypothesis (mean = 0)
            if xstd > 1e-10 and K > 1:
                t_stat, p_val = stats.ttest_1samp(xV, 0)
                neuron_p_values.append(p_val)
                stats_dict['tested'] += 1
            else:
                neuron_p_values.append(1.0)  # Non-significant p-value for low variance
                stats_dict['low_variance'] += 1
        
        # Apply FDR correction for this neuron's edges only
        if len(neuron_p_values) > 0:
            stats_dict['neurons_tested'] += 1
            
            # Count testable edges for this neuron
            testable_edges = sum(1 for p in neuron_p_values if p < 1.0)
            
            if testable_edges > 0:
                # Stage 1: FDR correction
                rejected, p_adj, alpha_sidak, alpha_bonf = multipletests(
                    neuron_p_values, alpha=alpha, method='fdr_bh'
                )
                
                # Stage 2: Compute minimum effect size for this neuron
                neuron_means_array = np.array(neuron_means)
                fdr_significant_mask = np.array(rejected)
                
                if np.any(fdr_significant_mask):
                    # Get effect sizes of FDR-significant edges for this neuron
                    significant_effects = np.abs(neuron_means_array[fdr_significant_mask])
                    
                    if len(significant_effects) > 3:  # Need enough edges for percentile
                        # Use median + factor*MAD as minimum effect size for this neuron
                        median_effect = np.median(significant_effects)
                        mad_effect = np.median(np.abs(significant_effects - median_effect))
                        min_effect_size = median_effect + effect_size_factor * mad_effect
                    else:
                        # For few edges, use 75th percentile
                        min_effect_size = np.percentile(significant_effects, 75) if len(significant_effects) > 1 else significant_effects[0] * 0.8
                    
                    if verb >= 3:
                        print(f"  Neuron {neuron_i}: auto effect size threshold = {min_effect_size:.4f}")
                else:
                    min_effect_size = 0.0  # No significant edges to filter
                
                neuron_fdr_count = 0
                neuron_final_count = 0
                
                # Apply both FDR and effect size filters
                for k, (i, j) in enumerate(neuron_positions):
                    if rejected[k]:  # Passed FDR
                        stats_dict['significant_fdr'] += 1
                        neuron_fdr_count += 1
                        
                        # Check effect size
                        if abs(neuron_means[k]) >= min_effect_size:
                            W_edges_mean[i, j] = neuron_means[k]
                            W_edges_std[i, j] = neuron_stds[k]
                            stats_dict['significant'] += 1
                            neuron_final_count += 1
                            
                            # Count by sign
                            if neuron_means[k] >= 0:
                                stats_dict['acc_positive'] += 1
                            else:
                                stats_dict['acc_negative'] += 1
                            
                            if verb >= 3:
                                print(f"  Final edge ({i},{j}): mean={neuron_means[k]:.4f}, std={neuron_stds[k]:.4f}, p_adj={p_adj[k]:.4e}")
                        else:
                            stats_dict['effect_filtered'] += 1
                
                if neuron_final_count > 0:
                    stats_dict['neurons_with_edges'] += 1
                    
                if verb >= 3:
                    print(f"  Neuron {neuron_i}: {testable_edges} testable, {neuron_fdr_count} FDR-sig, {neuron_final_count} final")
    
    if verb >= 1:
        print("RESULTS:")
        print(f"  Total off-diagonal elements: {stats_dict['offdiag']}")
        print(f"  Testable (sufficient var):   {stats_dict['tested']} ({100*stats_dict['tested']/stats_dict['offdiag']:.1f}%)")
        print(f"  Low variance (skipped):      {stats_dict['low_variance']} ({100*stats_dict['low_variance']/stats_dict['offdiag']:.1f}%)")
        print(f"  Significant (FDR only):      {stats_dict['significant_fdr']} ({100*stats_dict['significant_fdr']/stats_dict['offdiag']:.1f}%)")
        print(f"  Effect size filtered:        {stats_dict['effect_filtered']} ({100*stats_dict['effect_filtered']/stats_dict['offdiag']:.1f}%)")
        print(f"  Accepted (positive):         {stats_dict['acc_positive']} ({100*stats_dict['acc_positive']/stats_dict['offdiag']:.1f}%)")
        print(f"  Accepted (negative):         {stats_dict['acc_negative']} ({100*stats_dict['acc_negative']/stats_dict['offdiag']:.1f}%)")
        print(f"  TOTAL SELECTED:              {stats_dict['significant']}/{stats_dict['offdiag']} ({100*stats_dict['significant']/stats_dict['offdiag']:.1f}%)")
        print(f"  Neurons tested:              {stats_dict['neurons_tested']} out of {M}")
        print(f"  Neurons with edges:          {stats_dict['neurons_with_edges']} ({100*stats_dict['neurons_with_edges']/max(1,stats_dict['neurons_tested']):.1f}% of tested)")
        
        # Verification
        print(f"  Verification: {stats_dict['offdiag']} = {stats_dict['tested']} testable + {stats_dict['low_variance']} low-var")
        print()
    
    return W_edges_mean, W_edges_std, stats_dict
