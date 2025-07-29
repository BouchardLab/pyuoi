#!/usr/bin/env python3
"""
Evaluation and plotting script for analyzing results from fit_poisson.py.

This script:
1. Loads the learned A & B matrices from fit_poisson.py
2. Performs edge selection based on target_sparsity
3. Evaluates against ground truth
4. Generates plots

Usage:
    python eval1_fitPoisson.py --dataName <name> --dataPath <path> --showPlots <plot_types> --target_sparsity <sparsity>

Example:
    python eval1_fitPoisson.py --dataName test_dale --dataPath out/ --showPlots abc --target_sparsity 0.8
"""

import numpy as np
import os
import argparse
import sys
from PlotterFitPoissonV1 import Plotter
from UtilFitPoisson import load_ground_truth, analyze_edge_detection, save_evaluation_results

def load_fit_results(dataName, dataPath):
    """
    Load all results from fit_poisson.py output files.
    
    Args:
        dataName: Base name for the dataset
        dataPath: Path to the data directory
        
    Returns:
        dict: Dictionary containing all loaded data
    """
    print(f"Loading fit results for {dataName} from {dataPath}")
    
    # Construct file paths
    struct_file = os.path.join(dataPath, f"{dataName}.struct.npy")
    truth_file = os.path.join(dataPath, f"{dataName}.truth.npz")
    eval_file = os.path.join(dataPath, f"{dataName}_struct_eval.npz")
    
    # Load structure results
    if not os.path.exists(struct_file):
        raise FileNotFoundError(f"Structure file not found: {struct_file}")
    
    print(f"Loading structure results from {struct_file}")
    struct_data = np.load(struct_file, allow_pickle=True).item()
    A_stage1 = struct_data['A_stage1']
    B_stage1 = struct_data['B_stage1']
    train_losses = struct_data['train_losses']
    val_losses = struct_data['val_losses']
    firing_rates = struct_data['firing_rates']
    
    print(f"Loaded structure data: A_stage1 shape={A_stage1.shape}")
    print(f"Training losses: {len(train_losses)} epochs")
    
    # Load ground truth if available
    A_true = None
    B_true = None
    num_excite = 0
    edge_analysis = None
    
    if os.path.exists(truth_file):
        print(f"Loading ground truth from {truth_file}")
        A_true, B_true, conf, num_excite = load_ground_truth(truth_file)
        print(f"Loaded ground truth: A_true shape={A_true.shape}, num_excite={num_excite}")
    else:
        print("Warning: Ground truth file not found. Evaluation plots will be disabled.")
    
    # Load evaluation results if available
    if os.path.exists(eval_file):
        print(f"Loading evaluation results from {eval_file}")
        eval_data = np.load(eval_file, allow_pickle=True)
        print(f"Loaded evaluation data with keys: {list(eval_data.keys())}")
    
    return {
        'A_true': A_true,
        'B_true': B_true,
        'A_stage1': A_stage1,
        'B_stage1': B_stage1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'firing_rates': firing_rates,
        'edge_analysis': edge_analysis,
        'num_excite': num_excite
    }

def perform_edge_selection(A_stage1, target_sparsity, n_neurons):
    """
    Perform edge selection based on target sparsity.
    
    Args:
        A_stage1: Learned weight matrix
        target_sparsity: Target sparsity level (0.9 = 90% zeros)
        n_neurons: Number of neurons
        
    Returns:
        dict: Dictionary containing mask and analysis results
    """
    print(f"\n=== Edge Selection (Target Sparsity: {target_sparsity*100:.1f}%) ===")
    
    # Standard thresholding
    A_abs = np.abs(A_stage1)
    n_total = n_neurons**2
    n_keep = int(n_total * (1 - target_sparsity))
    
    # Create diagonal mask
    diag_mask = np.eye(n_neurons, dtype=bool)
    off_diag_mask = ~diag_mask
    
    # Always include diagonal elements in the mask
    mask = diag_mask.copy()
    
    # For off-diagonal elements, find threshold that keeps top n_keep_off_diag connections
    n_off_diag = n_neurons * (n_neurons - 1)  # Number of off-diagonal elements
    n_keep_off_diag = n_keep - n_neurons  # Reserve n_neurons spots for diagonal elements
    
    if n_keep_off_diag > 0:
        # Get sorted off-diagonal weights (descending order)
        off_diag_weights = A_abs[off_diag_mask]
        sorted_off_diag = np.sort(off_diag_weights)[::-1]
        
        if n_keep_off_diag < len(sorted_off_diag):
            adaptive_threshold = sorted_off_diag[n_keep_off_diag-1]
        else:
            adaptive_threshold = 0.0
        
        # Apply threshold to off-diagonal elements
        off_diag_selected = A_abs >= adaptive_threshold
        mask = mask | off_diag_selected
    else:
        # If n_keep is less than n_neurons, only keep diagonal elements
        adaptive_threshold = 0.0
        print(f"Warning: n_keep ({n_keep}) < n_neurons ({n_neurons}), keeping only diagonal elements")
    
    actual_sparsity = 1 - np.sum(mask) / n_total
    print(f"Target sparsity: {target_sparsity*100:.1f}%, Achieved sparsity: {actual_sparsity*100:.1f}%")
    print(f"Adaptive threshold: {adaptive_threshold:.6f}")
    print(f"Identified {np.sum(mask)} non-zero connections out of {n_total} possible")
    print(f"  - Diagonal elements: {np.sum(diag_mask)} (always included)")
    print(f"  - Off-diagonal elements: {np.sum(mask & off_diag_mask)}")
    
    # Analyze detected connections by neuron index
    print(f"\nAnalyzing detected connections by neuron index:")
    outgoing_connections = np.sum(mask, axis=1)  # Total connections FROM each neuron
    incoming_connections = np.sum(mask, axis=0)  # Total connections TO each neuron
    
    for i in range(0, n_neurons, 5):  # Print every 5th neuron
        end_idx = min(i + 5, n_neurons)
        out_slice = outgoing_connections[i:end_idx]
        in_slice = incoming_connections[i:end_idx]
        conn_pairs = [f"{in_val:2d}/{out_val:2d}" for in_val, out_val in zip(in_slice, out_slice)]
        print(f"  Neurons {i:2d}-{end_idx-1:2d}: in/out: {conn_pairs}")
    
    # Check weight magnitudes by neuron region
    print(f"\nWeight magnitudes by neuron region:")
    for region_start in [0, 10, 20, 30]:
        if region_start >= n_neurons:
            break
        region_end = min(region_start + 10, n_neurons)
        region_weights = A_abs[region_start:region_end, :]
        max_weight = np.max(region_weights)
        mean_weight = np.mean(region_weights[region_weights > 0]) if np.any(region_weights > 0) else 0
        print(f"  Neurons {region_start:2d}-{region_end-1:2d}: max={max_weight:.6f}, mean={mean_weight:.6f}")
    
    return {
        'mask': mask,
        'target_sparsity': target_sparsity,
        'actual_sparsity': actual_sparsity,
        'adaptive_threshold': adaptive_threshold
    }

def create_metadata(dataName, dataPath, bigD):
    """Create metadata dictionary for plotting."""
    # Get final losses
    train_losses = bigD['train_losses']
    val_losses = bigD['val_losses']
    final_train_loss = train_losses[-1] if len(train_losses) > 0 else -999
    final_val_loss = val_losses[-1] if len(val_losses) > 0 else -999
    
    # Get number of epochs
    num_epochs = len(train_losses) if len(train_losses) > 0 else -999
    
    # Load metadata from saved structure file
    num_samples = bigD.get('num_samples', -999)
    initial_lr = bigD.get('initial_lr', -0.222)
    train_time_min = bigD.get('train_time_min', -3)
    
    # Create metadata similar to what fit_poisson.py creates
    MD = {
        'num_excit_neur': bigD['num_excite'],
        'short_name': dataName,
        'num_samples': num_samples,
        'initial_lr': initial_lr,
        'train_time_min': train_time_min,
        'num_epochs': num_epochs,
        'final_loss': final_val_loss,  # Add final validation loss
        'final_train_loss': final_train_loss,  # Add final training loss
        'final_val_loss': final_val_loss  # Add final validation loss (alternative key)
    }
    
    return MD

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from fit_poisson.py")
    parser.add_argument("--dataName", type=str, default='dale_M50_3M', help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="out/", help="Path to the data directory")
    parser.add_argument("--target_sparsity", type=float, default=0.8, help="Target sparsity level (0.9 = 90% zeros)")
    parser.add_argument('-p',"--showPlots", type=str, default="abcd", help="Plot types to show: a=structure, b=distributions, c=reconstruction, d=category")
    parser.add_argument("--outPath", type=str, default="out/", help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level")
    parser.add_argument("--formatVenue", type=str, default="prod", help="Plot format venue")
    
    args = parser.parse_args()
    
    # Set output path
    if args.outPath is None:
        args.outPath = args.dataPath
    
    print("=== Evaluate Fit Results ===")
    print(f"Dataset: {args.dataName}")
    print(f"Data path: {args.dataPath}")
    print(f"Output path: {args.outPath}")
    print(f"Target sparsity: {args.target_sparsity*100:.1f}%")
    print(f"Show plots: {args.showPlots}")
    
    # Load all fit results
    bigD = load_fit_results(args.dataName, args.dataPath)
    
    # Perform edge selection
    edge_selection_results = perform_edge_selection(
        bigD['A_stage1'], args.target_sparsity, bigD['A_stage1'].shape[0]
    )
    
    # Add edge selection results to bigD
    bigD.update(edge_selection_results)
    
    # Evaluate against ground truth if available
    if bigD['A_true'] is not None:
        print(f"\n=== Evaluation Against Ground Truth ===")
        edge_analysis = analyze_edge_detection(bigD['A_true'], bigD['mask'], bigD['num_excite'])
        bigD['edge_analysis'] = edge_analysis
        
        # Save evaluation results
        eval_file = os.path.join(args.dataPath, f"{args.dataName}_struct_eval.npz")
        save_evaluation_results(eval_file, edge_analysis, bigD['num_excite'])
        print(f"Evaluation results saved to {eval_file}")
    
    # Create metadata
    MD = create_metadata(args.dataName, args.dataPath, bigD)
    
    # Setup plotter
    args.prjName = args.dataName + '_struct'
    plot = Plotter(args)
    
    # Generate requested plots
    print(f"\n=== Generating Plots ===")
    
    if 'a' in args.showPlots:
        print("Generating structure results plot (a)")
        plot.structure_results(bigD, MD, figId=1)
    
    if 'b' in args.showPlots:
        print("Generating weight distributions plot (b)")
        plot.weight_distributions(bigD, MD, figId=2)
    
    if 'c' in args.showPlots:
        print("Generating weight reconstruction results plot (c)")
        plot.weight_reconstruction_results(bigD, MD, figId=3)
    
    if 'd' in args.showPlots:
        print("Generating weight category analysis plot (d)")
        plot.weight_category_analysis(bigD, MD, figId=4)
    
    # Display all plots
    plot.display_all()
    
    print("\n=== Evaluation Complete ===")
    print(f"Plots saved to: {args.outPath}")
    print(f"Identified {np.sum(bigD['mask'])} connections out of {bigD['A_stage1'].shape[0]**2} possible")

if __name__ == "__main__":
    main() 
