#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Evaluation Functions for Neural Connectivity Inference

This module provides comprehensive evaluation and reporting functions for analyzing
the performance of neural connectivity inference algorithms against ground truth
connectivity matrices. It focuses on sparse graph evaluation with Dale's principle
structure (excitatory/inhibitory neuron types).

Key Evaluation Categories:
1. Connection-Level Analysis:
   - Excitatory off-diagonal connections (positive weights)
   - Inhibitory off-diagonal connections (negative weights)
   - Classification metrics: precision, recall, F1-score, TP/FP/TN/FN counts

2. Neuron-Level Analysis:
   - Row-level classification of neuron types (excitatory vs inhibitory)
   - Based on majority connection type from each neuron

3. Confidence Analysis:
   - Posterior probability evaluation for connection signs
   - Confidence matrices for correct vs incorrect predictions
   - Uncertainty quantification for edge predictions

Key Functions:
- evaluate_connectivity(): Computes comprehensive classification metrics
- print_evaluation_report(): Formatted performance reporting
- _print_matrix(): Matrix visualization with error highlighting
- print_confidence_by_correctness(): Confidence analysis for TP/FP edges
- save_and_report_results(): Complete evaluation pipeline

Matrix Printing Features:
- Error highlighting with '*' for incorrectly predicted connections
- Per-row connection counts for sparsity analysis
- Confidence matrices showing only predicted edges
- Formatted output for easy interpretation

Used by: bayes_sparse_regression.py for Bayesian model evaluation
"""

import numpy as np
import torch

# ==================================
#  CORE EVALUATION FUNCTIONS
# ==================================

def _compute_classification_metrics(pred_mask, true_mask, total_items):
    """Computes a full suite of classification metrics."""
    tp = (pred_mask & true_mask).sum().item()
    fp = (pred_mask & ~true_mask).sum().item()
    fn = (~pred_mask & true_mask).sum().item()
    tn = total_items - tp - fp - fn

    pred_true = tp + fp
    true_true = tp + fn

    precision = tp / pred_true if pred_true > 0 else 0.
    recall = tp / true_true if true_true > 0 else 0.
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.
    
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1_score": f1,
        "pred_true": pred_true, "true_true": true_true
    }

def evaluate_connectivity(inferred_T, truth_E, mask):
    stats = {}
    offdiag_T = inferred_T[mask]
    offdiag_E = truth_E[mask]
    num_offdiag_items = len(offdiag_T)
    stats['excitatory'] = _compute_classification_metrics(pred_mask=(offdiag_T > 0), true_mask=(offdiag_E > 0), total_items=num_offdiag_items)
    stats['inhibitory'] = _compute_classification_metrics(pred_mask=(offdiag_T < 0), true_mask=(offdiag_E < 0), total_items=num_offdiag_items)
    
    # Row-level excitatory/inhibitory binary classification
    M = inferred_T.shape[0]
    mask_2d = mask.view(M, M)
    
    # For each row, determine if it's excitatory (True) or inhibitory (False)
    pred_row_type = []
    true_row_type = []
    
    for i in range(M):
        # Get off-diagonal elements for this row
        row_mask = mask_2d[i, :]
        pred_row = inferred_T[i, :][row_mask]
        true_row = truth_E[i, :][row_mask]
        
        # Count excitatory and inhibitory connections in this row
        pred_exc_count = (pred_row > 0).sum().item()
        pred_inh_count = (pred_row < 0).sum().item()
        true_exc_count = (true_row > 0).sum().item()
        true_inh_count = (true_row < 0).sum().item()
        
        # Binary classification: True if excitatory (more exc than inh), False if inhibitory
        pred_row_type.append(pred_exc_count > pred_inh_count)
        true_row_type.append(true_exc_count > true_inh_count)
    
    pred_row_type = torch.tensor(pred_row_type)
    true_row_type = torch.tensor(true_row_type)
    
    stats['row_type'] = _compute_classification_metrics(pred_mask=pred_row_type, true_mask=true_row_type, total_items=M)
    
    return stats

def print_evaluation_report(stats, args, used_sparsity):
    print(f"\n--- Evaluation vs. Ground Truth (E) for '{args.input}.npz' after {args.num_epochs} epochs, prob_thresh={args.edge_probability:.3f}, actual_sparsity={used_sparsity:.3f} ---")
    
    def print_metrics(title, metrics, unit="edges"):
        print(f"  --- {title} ---")
        if unit == "edges":
            sparsity_info = f"Existing edges (pred/true): {metrics['pred_true']}/{metrics['true_true']}"
        else:  # unit == "rows"
            sparsity_info = f"Excitatory rows (pred/true): {metrics['pred_true']}/{metrics['true_true']}"
        print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}   - {sparsity_info}")
        print(f"    Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")

    print_metrics("Excitatory Off-Diagonal", stats['excitatory'], "edges")
    print_metrics("Inhibitory Off-Diagonal", stats['inhibitory'], "edges")
    print_metrics("Row-Level Excitatory/Inhibitory Classification", stats['row_type'], "rows")
    print("--------------------------------------------------------------------------------\n")

# ==================================
#  MATRIX PRINTING AND VISUALIZATION
# ==================================

def _print_matrix(name, matrix_np, mask_np, truth_matrix_np=None):
    """Helper to print a matrix with per-row non-zero counts.
    If truth_matrix_np is provided, it adds a '*' to wrongly predicted elements.
    """
    print(f"--- {name} ---")
    off_diag_counts = np.count_nonzero(matrix_np[:,] * mask_np[:,], axis=1)

    if truth_matrix_np is None:
        # Simple print for matrices without comparison (like the ground truth E)
        for i, row in enumerate(matrix_np):
            # Format each element to be right-aligned in a 3-char space
            formatted_row = " ".join([f"{val:3d}" for val in row])
            print(f"{i:3d}: ({off_diag_counts[i]:3d}) [ {formatted_row} ]")
        return

    # Advanced print with '*' for wrongly predicted edges
    truth_sign_matrix = np.sign(truth_matrix_np)
    str_matrix = np.empty_like(matrix_np, dtype=object)

    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            val = matrix_np[i, j]
            if np.sign(val) != truth_sign_matrix[i, j]:
                str_matrix[i, j] = f"{val}*"
            else:
                str_matrix[i, j] = f"{val}"
    
    # Custom print to avoid apostrophes and control alignment
    for i, row in enumerate(str_matrix):
        # Format each element to be right-aligned in a 3-char space
        formatted_row = " ".join([f"{s:>3}" for s in row])
        print(f"{i:3d}: ({off_diag_counts[i]:3d}) [ {formatted_row} ]")

def _print_confidence_matrix(name, confidence_np, W_est_np=None, precision=2):
    """Helper to print the confidence matrix. If W_est_np is provided, shows confidence only for found edges."""
    print(f"--- {name} (Posterior Probability of Sign) ---")
    printable_conf = confidence_np.copy()
    if W_est_np is not None:
        printable_conf[W_est_np == 0] = 0.
    
    # Custom print to guarantee alignment and formatting
    for row in printable_conf:
        # Format each number to a fixed width (e.g., 6 chars) and precision
        # Smartly print zeros to reduce noise
        formatted_parts = []
        for num in row:
            if np.isclose(num, 0):
                formatted_parts.append(f"{'0.':>6}")
            else:
                formatted_parts.append(f"{num:6.{precision}f}")
        formatted_row = "".join(formatted_parts)
        print(f"[{formatted_row} ]")

def print_confidence_by_correctness(W_est, W_confidence, truth_E, mask):
    """Creates and prints confidence tables for correctly (TP) and incorrectly (FP) identified off-diagonal edges."""
    W_est_np = W_est.cpu().numpy().astype(int)
    W_conf_np = W_confidence.cpu().numpy()
    truth_E_np = truth_E.cpu().numpy().astype(int)
    mask_np = mask.cpu().numpy()

    # --- Off-Diagonal Confidence Evaluation ---
    pred_mask_flat = W_est_np[mask_np] != 0
    true_mask_flat = truth_E_np[mask_np] != 0
    tp_mask_flat = pred_mask_flat & true_mask_flat
    fp_mask_flat = pred_mask_flat & ~true_mask_flat
    conf_flat = W_conf_np[mask_np]
    
    correct_conf_matrix = np.zeros_like(W_conf_np)
    correct_conf_matrix[mask_np] = conf_flat * tp_mask_flat
    
    wrong_conf_matrix = np.zeros_like(W_conf_np)
    wrong_conf_matrix[mask_np] = conf_flat * fp_mask_flat

    # Keep diagonal elements as zero in confidence matrices (not evaluated)
    np.fill_diagonal(correct_conf_matrix, 0.0)
    np.fill_diagonal(wrong_conf_matrix, 0.0)

    _print_confidence_matrix("Confidence for CORRECT Edges (TPs)", correct_conf_matrix)
    _print_confidence_matrix("Confidence for WRONG Edges (FPs)", wrong_conf_matrix, precision=2)

# ==================================
#  RESULTS SAVING AND REPORTING
# ==================================

def save_and_report_results(W_est, W_confidence, truth_E, mask, train_stats, args, device, used_sparsity):
    W_est_np = W_est.cpu().numpy().astype(int)

    # Do the evaluation just before the final report
    if truth_E is not None and args.verb > 0:
        stats = evaluate_connectivity(W_est, truth_E, mask)
        print_evaluation_report(stats, args, used_sparsity)

    if args.verb > 0:
        num_nonzero = np.count_nonzero(W_est_np[mask.cpu().numpy()])
        print(f"Number of non-zero weights, excluding diagonal: {num_nonzero}")

        if args.verb > 1:
            mask_np = mask.cpu().numpy()
            truth_E_np = truth_E.cpu().numpy().astype(int) if truth_E is not None else None
            _print_matrix("Final Estimated Matrix (T)", W_est_np, mask_np, truth_matrix_np=truth_E_np)
            
            if truth_E is not None:
                print_confidence_by_correctness(W_est, W_confidence, truth_E, mask)
            else:
                # Fallback for no ground truth
                W_conf_np = W_confidence.cpu().numpy()
                _print_confidence_matrix("Confidence for T", W_conf_np, W_est_np)
        
        print(f"Fitting took {train_stats['elapsed_minutes']:.2f} minutes")
        if device.type == 'cuda':
            print(f"GPU stats mid-training: temp={train_stats['gpu_temp']}C, power={train_stats['gpu_power']}W, usage={train_stats['gpu_util']}%")
    
    out_fname = "model/W_est.npy"
    np.save(out_fname, W_est_np)
    if args.verb > 0:
        print(f"Saved estimated ±1/0 adjacency to {out_fname}") 