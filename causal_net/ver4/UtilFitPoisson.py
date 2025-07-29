#!/usr/bin/env python3
"""
Utility functions for Poisson GLM fitting.
Contains data loading and post-processing functions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import time

def load_data_and_rates(args):
    """
    Load spike data and firing rates from files.
    
    Args:
        args: Command line arguments containing dataName, dataPath, and num_samples
        
    Returns:
        dict: Dictionary containing:
            - Y: Spike data array (time_steps x n_neurons)
            - firing_rates: Array of firing rates for each neuron
            - n_neurons: Number of neurons
            - n_samples: Number of time samples used
    """
    # Construct file paths
    spikes_file = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    
    # Load spike data and firing rates
    print(f"Loading spike data from {spikes_file}")
    spike_data = np.load(spikes_file, allow_pickle=True)
    Y = spike_data['Y']
    firing_rates = spike_data['firing_rates']
    
    # Bracket (clip) firing rates to reasonable bounds
    min_rate = 0.01  # Minimum 0.01 Hz
    max_rate = 100.0  # Maximum 100 Hz
    firing_rates_original = firing_rates.copy()
    firing_rates = np.clip(firing_rates, min_rate, max_rate)
    
    # Report bracketing statistics
    n_clipped = np.sum((firing_rates_original < min_rate) | (firing_rates_original > max_rate))
    if n_clipped > 0:
        print(f"Bracketed {n_clipped} firing rates to range [{min_rate}, {max_rate}] Hz")
        print(f"  Original range: [{np.min(firing_rates_original):.3f}, {np.max(firing_rates_original):.3f}] Hz")
        print(f"  Bracketed range: [{np.min(firing_rates):.3f}, {np.max(firing_rates):.3f}] Hz")
    else:
        print(f"All firing rates within reasonable bounds [{min_rate}, {max_rate}] Hz")
    
    # Determine number of neurons
    n_neurons = Y.shape[1]
    original_steps = Y.shape[0]
    print(f"Loaded {original_steps} time steps for {n_neurons} neurons")
    print(f"Loaded firing rates: shape={firing_rates.shape}")
    
    # Clip data if num_samples is specified
    if args.num_samples is not None and args.num_samples > 0:
        if args.num_samples > original_steps:
            print(f"Warning: Requested {args.num_samples} samples but only {original_steps} available. Using all available data.")
            args.num_samples = original_steps
        else:
            Y = Y[:args.num_samples, :]
            print(f"Clipped data to {args.num_samples} time samples (from {original_steps})")
    
    n_samples = Y.shape[0]
    print(f"Using {n_samples} time steps for fitting")
    
    return {
        'Y': Y,
        'firing_rates': firing_rates,
        'n_neurons': n_neurons,
        'n_samples': n_samples,
        'coincidence_rates':  spike_data['coincidence_rates']
    }

def preprocess_data_blind(Y):
    """Preprocess spike data without knowledge of excitatory/inhibitory identity."""
    print("\n=== Data Preprocessing (Blind) ===")
    print(f"Data shape: {Y.shape}")
    print(f"Total time steps: {Y.shape[0]}")
    n_samples = Y.shape[0] - 1
    print(f"Number of neurons: {Y.shape[1]}")
    print(f"Total number of training samples (time steps - 1): {n_samples}")
    
    # Calculate sparsity of data
    sparsity = 1 - np.count_nonzero(Y) / Y.size
    print(f"\nData sparsity: {sparsity*100:.1f}% zeros")
    
    return None

def create_data_loaders(Y_tensor, batch_size=256, num_workers=4):
    """
    Create train and validation data loaders from spike tensor.
    
    Args:
        Y_tensor: PyTorch tensor of shape (time_steps, n_neurons)
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create dataset
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Optimized DataLoader settings for GPU utilization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=min(num_workers, 8), pin_memory=True, 
                             persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=min(num_workers, 4), pin_memory=True,
                           persistent_workers=True, prefetch_factor=2)
    
    print(f"DataLoader optimization: train_workers={min(num_workers, 8)}, val_workers={min(num_workers, 4)}, prefetch=4/2")
    
    return train_loader, val_loader

def analyze_edge_detection(A_true, mask_detected, num_excite):
    """Analyze edge detection performance separately for excitatory and inhibitory connections."""
    
    # Create true edge mask (non-zero elements) - exclude diagonal
    mask_true = np.abs(A_true) > 1e-10
    diag_mask = np.eye(mask_true.shape[0], dtype=bool)
    mask_true[diag_mask] = False  # Exclude diagonal elements from evaluation
    
    # Create masks for excitatory and inhibitory neurons (rows)
    exc_mask = np.zeros_like(mask_true)
    exc_mask[:num_excite, :] = True
    inh_mask = np.zeros_like(mask_true)
    inh_mask[num_excite:, :] = True
    
    # Exclude diagonal from neuron type masks
    exc_mask[diag_mask] = False
    inh_mask[diag_mask] = False
    
    def calculate_metrics(true_edges, detected_edges):
        """Calculate TP, FP, TN, FN and derived metrics."""
        tp = np.sum(true_edges & detected_edges)
        fp = np.sum(~true_edges & detected_edges)
        tn = np.sum(~true_edges & ~detected_edges)
        fn = np.sum(true_edges & ~detected_edges)
        
        # Derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        
        return {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1_score': f1_score,
            'specificity': specificity, 'accuracy': accuracy
        }
    
    # Overall analysis (off-diagonal only)
    overall_metrics = calculate_metrics(mask_true, mask_detected)
    
    # Excitatory analysis (excitatory rows, off-diagonal only)
    exc_true = mask_true & exc_mask
    exc_detected = mask_detected & exc_mask
    exc_metrics = calculate_metrics(exc_true, exc_detected)
    
    # Debug: Print total excitatory weights for verification
    total_excit_weights = np.sum(np.abs(A_true[:num_excite, :]) > 1e-10)
    print(f"\nDebug: Total excitatory weights (threshold 1e-10): {total_excit_weights}")
    print(f"Debug: Excitatory true edges (off-diagonal): {np.sum(exc_true)}")
    print(f"Debug: Excitatory TP + FN = {exc_metrics['tp'] + exc_metrics['fn']}")
    
    # Inhibitory analysis (inhibitory rows, off-diagonal only)
    inh_true = mask_true & inh_mask
    inh_detected = mask_detected & inh_mask
    inh_metrics = calculate_metrics(inh_true, inh_detected)
    
    # Print results
    print("\n=== Edge Detection Analysis (Off-Diagonal Only) ===")
    print("Overall (off-diagonal):")
    print(f"  TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}, TN: {overall_metrics['tn']}, FN: {overall_metrics['fn']}")
    print(f"  Precision: {overall_metrics['precision']:.3f}, Recall: {overall_metrics['recall']:.3f}")
    print(f"  Accuracy: {overall_metrics['accuracy']:.3f}, F1 Score: {overall_metrics['f1_score']:.3f}")
    
    print("\nExcitatory rows (off-diagonal):")
    print(f"  TP: {exc_metrics['tp']}, FP: {exc_metrics['fp']}, TN: {exc_metrics['tn']}, FN: {exc_metrics['fn']}")
    print(f"  Precision: {exc_metrics['precision']:.3f}, Recall: {exc_metrics['recall']:.3f}")
    print(f"  Accuracy: {exc_metrics['accuracy']:.3f}, F1 Score: {exc_metrics['f1_score']:.3f}")
    
    print("\nInhibitory rows (off-diagonal):")
    print(f"  TP: {inh_metrics['tp']}, FP: {inh_metrics['fp']}, TN: {inh_metrics['tn']}, FN: {inh_metrics['fn']}")
    print(f"  Precision: {inh_metrics['precision']:.3f}, Recall: {inh_metrics['recall']:.3f}")
    print(f"  Accuracy: {inh_metrics['accuracy']:.3f}, F1 Score: {inh_metrics['f1_score']:.3f}")
    
    # Create the true masks for plotting
    diag_mask = np.eye(mask_true.shape[0], dtype=bool)
    
    # True masks for different categories
    true_masks = {
        'diagonal': diag_mask,
        'excitatory': exc_mask,
        'inhibitory': inh_mask
    }
    
    return {
        'overall': overall_metrics,
        'excitatory': exc_metrics,
        'inhibitory': inh_metrics,
        'true_masks': true_masks
    }

def save_structure_results(struct_file, A_stage1, B_stage1, train_losses, val_losses, firing_rates, args, metadata):
    """Save structure identification results to file."""
    save_dict = {
        'A_stage1': A_stage1,
        'B_stage1': B_stage1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'firing_rates': firing_rates,
        'args': vars(args)
    }
    
    # Add metadata if provided

    save_dict.update(metadata)
    
    np.save(struct_file, save_dict)
    print(f"\nStructure results saved to {struct_file}")

def save_evaluation_results(eval_file, edge_analysis, num_excite):
    """Save evaluation results to file."""
    eval_dict = {
        'edge_analysis': edge_analysis,
        'num_excite': num_excite
    }
    np.savez(eval_file, **eval_dict)
    print(f"Evaluation results saved to {eval_file}")

def XXload_ground_truth(truth_file):
    """Load ground truth data for evaluation."""
    print(f"\n=== Loading Ground Truth for Evaluation ===")
    print(f"Loading ground truth from {truth_file}")
    truth_data = np.load(truth_file, allow_pickle=True)
    A_true = truth_data['A']
    B_true = truth_data['B_intercept']
    conf = truth_data['conf'].item()
    num_excite = conf['num_excite']# Load spike data and firing rates
    print(f"Loading spike data from {spikes_file}")


    
    return A_true, B_true, conf, num_excite 
