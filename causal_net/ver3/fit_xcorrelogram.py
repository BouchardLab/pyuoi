#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
ingle file contains the complete implementation with:
Data loading from HDF5 files
Two cross-correlogram methods: basic GPU and FFT-accelerated with CuPy
Statistical significance testing with shuffle controls
Network inference with sparsity constraints
Neuron classification (excitatory/inhibitory)
Results saving to HDF5
Example usage with error handling

IMG=nersc/pytorch:25.02.01
 salloc -q interactive -C gpu --image $IMG  -t 4:00:00 -A m2043 -N 1
 shifter bash
'''

import torch
import h5py
import numpy as np
from scipy import sparse
try:
    import cupy as cp
    from cupyx.scipy import signal
    CUPY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    cp = None
    signal = None # so linter does not complain
    CUPY_AVAILABLE = False
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for compute nodes
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import argparse
import sys
import os

class SpikeDataLoader:
    """Load and preprocess spike data from HDF5 file"""
    def __init__(self, hdf5_path: str, device: str = 'cuda', clip_bins: int = None, max_lag: int = 20):
        with h5py.File(hdf5_path, 'r') as f:
            spike_data = f['spikes_data'][:]  # (40, 299950)

            # Load ground truth if available
            if 'true_network_matrix' in f:
                self.true_network_matrix = f['true_network_matrix'][:].T
                print("Found ground truth network matrix.")
                # Add diagnostic print, ignoring diagonal
                gt_no_diag = self.true_network_matrix.copy()
                np.fill_diagonal(gt_no_diag, 0)
                gt_excitatory_count = np.sum(gt_no_diag > 0)
                gt_inhibitory_count = np.sum(gt_no_diag < 0)
                print(f"  - Diagnostic: GT matrix contains {gt_excitatory_count} non-diagonal excitatory and {gt_inhibitory_count} non-diagonal inhibitory connections.")
            else:
                self.true_network_matrix = None
            

        if clip_bins is not None and clip_bins < spike_data.shape[1]:
            print(f"Clipping time bins from {spike_data.shape[1]} to {clip_bins}")
            spike_data = spike_data[:, :clip_bins]

        print(f"Using input data with shape: {spike_data.shape} (neurons, time_bins)")
        
        # Convert to sparse format for efficiency
        # Store as list of spike times per neuron
        self.spike_times = []
        for neuron_idx in range(spike_data.shape[0]):
            spike_indices = np.where(spike_data[neuron_idx])[0]
            self.spike_times.append(torch.tensor(spike_indices, device=device))
        
        self.n_neurons = len(self.spike_times)
        self.n_bins = spike_data.shape[1]
        self.device = device
        self.spike_data_dense = spike_data  # Keep for CuPy operations

        # Print detailed statistics after tensors are created
        self._print_firing_pattern_statistics()
        self._print_interaction_statistics(max_lag)

    def _print_firing_pattern_statistics(self):
        """Prints statistics about single neuron firing patterns (ISIs)."""
        
        neuron_avg_isis = []
        for spikes in self.spike_times:
            if len(spikes) > 1:
                isis = torch.diff(spikes)
                neuron_avg_isis.append(isis.float().mean().item())

        print("\n--- Single Neuron Firing Statistics ---")
        if neuron_avg_isis:
            neuron_avg_isis_np = np.array(neuron_avg_isis)
            
            print(f"  Metrics for avg. time between spikes (ISI) across all firing neurons:")
            print(f"    - Overall Mean of neuron avgs: {np.mean(neuron_avg_isis_np):.2f} bins")
            print(f"    - Min of neuron avgs  : {np.min(neuron_avg_isis_np):.2f} bins")
            print(f"    - Median of neuron avgs: {np.median(neuron_avg_isis_np):.2f} bins")
            print(f"    - Max of neuron avgs  : {np.max(neuron_avg_isis_np):.2f} bins")
        else:
            print("  Not enough spikes to calculate inter-spike intervals.")
        print("---------------------------------------")

    def _print_interaction_statistics(self, max_lag: int):
        """Prints metrics about spike coincidences within a lag window."""
        total_spikes_with_neighbors = 0
        total_neighbor_spikes = 0
        total_neighbor_neurons = 0
        num_sampled_spikes = 0
        sampling_prob = 0.1

        total_spikes_overall = sum(len(st) for st in self.spike_times)

        if total_spikes_overall == 0:
            print("\nNo spikes found in data, skipping interaction statistics.")
            return

        # This can be slow, so we'll show progress
        print(f"\nCalculating spike interaction statistics (sampling {sampling_prob:.0%} of primary spikes)...")
        
        for i in range(self.n_neurons):
            print(f"  ... analyzing neuron {i+1}/{self.n_neurons}", end='\r')
            spikes_i = self.spike_times[i]
            if len(spikes_i) == 0:
                continue

            # Create a concatenated tensor of all *other* spikes and their neuron indices
            other_spikes_list = [self.spike_times[j] for j in range(self.n_neurons) if i != j]
            other_indices_list = [torch.full_like(self.spike_times[j], j) for j in range(self.n_neurons) if i != j]
            
            if not other_spikes_list:
                continue

            other_spikes = torch.cat(other_spikes_list)
            other_neuron_indices = torch.cat(other_indices_list)

            # For each spike in neuron i, find neighbors (with sampling)
            for t_i in spikes_i:
                # Pick primary spike with 10% probability
                if torch.rand(1).item() > sampling_prob:
                    continue
                num_sampled_spikes += 1

                diffs = other_spikes - t_i
                coincident_mask = (diffs >= -max_lag) & (diffs <= max_lag)
                
                num_coincident_spikes = coincident_mask.sum().item()
                
                if num_coincident_spikes > 0:
                    total_spikes_with_neighbors += 1
                    total_neighbor_spikes += num_coincident_spikes
                    
                    coincident_neuron_idxs = other_neuron_indices[coincident_mask]
                    total_neighbor_neurons += torch.unique(coincident_neuron_idxs).numel()

        print("\n\n--- Spike Interaction Statistics ---")
        if total_spikes_with_neighbors > 0:
            prob_interaction = total_spikes_with_neighbors / num_sampled_spikes if num_sampled_spikes > 0 else 0
            avg_coincident_spikes = total_neighbor_spikes / total_spikes_with_neighbors
            avg_neighbor_neurons = total_neighbor_neurons / total_spikes_with_neighbors
            
            print(f"  Window size (max_lag): +/- {max_lag} bins")
            print(f"  Based on a {sampling_prob:.0%} random sample ({num_sampled_spikes} spikes):")
            print(f"    - Est. probability of interaction: {prob_interaction:.2%}")
            print(f"    - Avg neighboring spikes per interactive spike: {avg_coincident_spikes:.2f}")
            print(f"    - Avg unique neighbor neurons per interaction: {avg_neighbor_neurons:.2f}")
        else:
            print(f"  No spike interactions found within the +/- {max_lag} bin window (based on a {sampling_prob:.0%} sample).")
        print("------------------------------------\n")

class CrossCorrelogramGPU:
    """Basic GPU-accelerated cross-correlogram computation"""
    def __init__(self, max_lag: int = 20, batch_size: int = 1000):
        self.max_lag = max_lag
        self.batch_size = batch_size
        
    def compute_xcorr_batch(self, spike_times_i: torch.Tensor, 
                           spike_times_j_batch: List[torch.Tensor], 
                           lag_bins: torch.Tensor) -> torch.Tensor:
        """Compute cross-correlogram for one neuron against a batch using GPU"""
        batch_size = len(spike_times_j_batch)
        xcorr = torch.zeros((batch_size, len(lag_bins)-1), device='cuda')
        
        if len(spike_times_i) == 0:
            return xcorr
            
        # For each spike in neuron i, find coincident spikes in batch
        for t_i in spike_times_i:
            # Vectorized difference computation
            for j_idx, spikes_j in enumerate(spike_times_j_batch):
                if len(spikes_j) > 0:
                    diffs = spikes_j - t_i
                    # Only keep differences within lag window
                    valid_mask = (diffs >= -self.max_lag) & (diffs <= self.max_lag)
                    valid_diffs = diffs[valid_mask]
                    
                    if len(valid_diffs) > 0:
                        # Histogram using torch.histc
                        hist = torch.histc(valid_diffs.float(), 
                                         bins=len(lag_bins)-1, 
                                         min=float(lag_bins[0]), 
                                         max=float(lag_bins[-1]))
                        xcorr[j_idx] += hist
        
        return xcorr
    
    def compute_all_pairs(self, spike_data_loader: SpikeDataLoader, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-correlograms for all neuron pairs using batching"""
        n_neurons = spike_data_loader.n_neurons
        lag_bins = torch.arange(-self.max_lag, self.max_lag+1, device='cuda')
        
        # Initialize output tensor
        all_xcorr = torch.zeros((n_neurons, n_neurons, len(lag_bins)-1), device='cuda')
        
        # Process in batches to optimize GPU usage
        for i in range(n_neurons):
            if verbose and i > 0 and i % 10 == 0:
                print(f"  ... processing neuron {i}/{n_neurons}", end='\r')
            
            spike_times_i = spike_data_loader.spike_times[i]
            
            # Process neurons j in batches
            for j_start in range(0, n_neurons, self.batch_size):
                j_end = min(j_start + self.batch_size, n_neurons)
                j_batch = spike_data_loader.spike_times[j_start:j_end]
                
                xcorr_batch = self.compute_xcorr_batch(spike_times_i, j_batch, lag_bins)
                all_xcorr[i, j_start:j_end] = xcorr_batch
        
        if verbose:
            print(f"  ... processing neuron {n_neurons}/{n_neurons} - done")
        return all_xcorr, lag_bins[:-1]

class FastCrossCorrelogramGPU:
    """Ultra-fast cross-correlation using FFT on GPU with CuPy"""
    def __init__(self, max_lag: int = 20):
        self.max_lag = max_lag
        
    def compute_all_pairs_fft(self, spike_data_loader: SpikeDataLoader, verbose: bool = True) -> np.ndarray:
        """Ultra-fast cross-correlation using FFT on GPU"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available, cannot use FFT-based method.")

        n_neurons = spike_data_loader.n_neurons
        n_bins = spike_data_loader.n_bins
        
        # Convert sparse spike trains to dense binary vectors on GPU
        spike_matrix = cp.zeros((n_neurons, n_bins), dtype=cp.float32)
        for i, spike_times in enumerate(spike_data_loader.spike_times):
            if len(spike_times) > 0:
                spike_matrix[i, spike_times.cpu().numpy()] = 1
        
        # Compute all pairwise cross-correlations using FFT
        xcorr_full = cp.zeros((n_neurons, n_neurons, 2*self.max_lag+1))
        
        for i in range(n_neurons):
            if verbose and i > 0 and i % 10 == 0:
                print(f"  ... FFT processing neuron {i}/{n_neurons}", end='\r')
            for j in range(n_neurons):
                if i != j:
                    # Use CuPy's correlate function (GPU accelerated)
                    correlation = signal.correlate(spike_matrix[j], spike_matrix[i], mode='same')
                    # Extract relevant lags
                    center = n_bins // 2
                    start = center - self.max_lag
                    end = center + self.max_lag + 1
                    
                    # Handle edge cases
                    if start >= 0 and end <= len(correlation):
                        xcorr_full[i, j] = correlation[start:end]
        
        if verbose:
            print(f"  ... FFT processing neuron {n_neurons}/{n_neurons} - done")
        return cp.asnumpy(xcorr_full)

class SignificanceTester:
    """Statistical significance testing using shuffle controls"""
    def __init__(self, n_shuffles: int = 100, alpha: float = 0.01, use_fft: bool = True):
        self.n_shuffles = n_shuffles
        self.alpha = alpha
        self.use_fft = use_fft
        if self.use_fft:
            print("Using FFT-based method for significance testing shuffles.")
        else:
            print("Using direct method for significance testing shuffles.")

    def shuffle_test_gpu(self, spike_times: List[torch.Tensor], 
                        xcorr_obs: torch.Tensor, 
                        n_bins: int,
                        causal_win_start: int = 5,
                        causal_win_end: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute significance using circular shifts"""
        n_neurons = len(spike_times)
        device = xcorr_obs.device
        n_lags = xcorr_obs.shape[2]
        max_lag = (n_lags - 1) // 2
        
        # Generate random shifts for all shuffles at once
        shifts = torch.randint(1, n_bins, (self.n_shuffles, n_neurons), device=device)
        
        # Store maximum values from null distribution
        null_max = torch.zeros((self.n_shuffles, n_neurons, n_neurons), device=device)
        
        # Instantiate the right computer for the shuffle test
        if self.use_fft:
            xcorr_computer_shuffle = FastCrossCorrelogramGPU(max_lag=max_lag)
        else:
            xcorr_computer_shuffle = CrossCorrelogramGPU(max_lag=max_lag)

        shuffle_start_time = time.time()
        for shuffle_idx in range(self.n_shuffles):
            if (shuffle_idx + 1) % 10 == 0:
                elapsed_time = time.time() - shuffle_start_time
                print(f"  ... shuffle {shuffle_idx + 1}/{self.n_shuffles} (elapsed: {elapsed_time:.1f}s)", end='\r')
            
            # Create shifted spike times
            shifted_loader = type('obj', (object,), {
                'spike_times': [],
                'n_neurons': n_neurons,
                'n_bins': n_bins,
                'device': device
            })()
            
            for i, spikes in enumerate(spike_times):
                if len(spikes) > 0:
                    shifted = (spikes + shifts[shuffle_idx, i]) % n_bins
                    shifted_loader.spike_times.append(shifted)
                else:
                    shifted_loader.spike_times.append(spikes)
            
            # Compute cross-correlogram for shuffled data
            if self.use_fft:
                xcorr_shuffle_np = xcorr_computer_shuffle.compute_all_pairs_fft(shifted_loader, verbose=False)
                xcorr_shuffle = torch.tensor(xcorr_shuffle_np, device=device)
            else:
                xcorr_shuffle, _ = xcorr_computer_shuffle.compute_all_pairs(shifted_loader, verbose=False)
            
            # Store maximum absolute deviation in causal window
            causal_start_idx = n_lags // 2 + causal_win_start
            # +1 because Python slicing is exclusive at the end
            causal_end_idx = n_lags // 2 + causal_win_end + 1
            if causal_end_idx > n_lags:
                causal_end_idx = n_lags
            
            # Handle cases where the causal window is invalid for the given max_lag
            if causal_start_idx >= causal_end_idx:
                if shuffle_idx == 0: # Print warning only once
                    print(f"\nWarning: max_lag ({max_lag}) is too small to define the inclusive causal window [{causal_win_start}-{causal_win_end}ms]. Significance testing in this window will be skipped.", file=sys.stderr)
                null_max[shuffle_idx] = torch.zeros((n_neurons, n_neurons), device=device)
                continue
            
            causal_slice = torch.abs(xcorr_shuffle[:, :, causal_start_idx:causal_end_idx])
            null_max[shuffle_idx] = torch.max(causal_slice, dim=2)[0]
        
        print(f"\n  ... all {self.n_shuffles} shuffles processed.")
        
        # Compute p-values
        # For each connection, count how many shuffles exceed observed
        observed_max = torch.max(
            torch.abs(xcorr_obs[:, :, causal_start_idx:causal_end_idx]), 
            dim=2
        )[0]
        
        p_values = torch.zeros((n_neurons, n_neurons), device=device)
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j:
                    p_values[i, j] = (null_max[:, i, j] >= observed_max[i, j]).float().mean()
        
        # Compute significance thresholds
        thresholds = torch.quantile(null_max, 1 - self.alpha, dim=0)
        
        return p_values, thresholds

class NetworkInference:
    """Infer network connectivity with sparsity constraints"""
    def __init__(self, sparsity: float = 0.15):
        self.sparsity = sparsity
        
    def infer_connectivity(self, xcorr: torch.Tensor, 
                          p_values: torch.Tensor, 
                          lag_bins: torch.Tensor,
                          alpha: float = 0.01,
                          causal_win_start: int = 5,
                          causal_win_end: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer directed connectivity with excitatory/inhibitory classification"""
        n_neurons = xcorr.shape[0]
        device = xcorr.device
        
        # Find peak/trough for each neuron pair in the causal window (positive lags)
        causal_window = (lag_bins >= causal_win_start) & (lag_bins <= causal_win_end)
        causal_indices = torch.where(causal_window)[0]
        
        # Compute peak heights and positions
        peak_heights = torch.zeros((n_neurons, n_neurons), device=device)
        peak_positions = torch.zeros((n_neurons, n_neurons), dtype=torch.long, device=device)
        
        # Compute baseline (activity at zero lag)
        zero_lag_idx = len(lag_bins) // 2
        
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and len(causal_indices) > 0:
                    causal_xcorr = xcorr[i, j, causal_indices]
                    baseline = xcorr[i, j, zero_lag_idx]
                    
                    # Find maximum deviation from baseline
                    deviations = causal_xcorr - baseline
                    
                    # Find both positive and negative peaks
                    max_pos = torch.max(deviations)
                    max_neg = torch.min(deviations)
                    
                    if torch.abs(max_pos) > torch.abs(max_neg):
                        peak_heights[i, j] = max_pos
                        peak_positions[i, j] = torch.argmax(deviations)
                    else:
                        peak_heights[i, j] = max_neg
                        peak_positions[i, j] = torch.argmin(deviations)
        
        # Apply significance threshold
        significant_mask = p_values < alpha
        peak_heights *= significant_mask.float()
        
        # Enforce sparsity
        abs_heights = torch.abs(peak_heights)
        threshold = torch.quantile(abs_heights[abs_heights > 0].flatten(), 1 - self.sparsity)
        sparse_connections = abs_heights > threshold
        
        # Classify neurons as excitatory or inhibitory
        neuron_types = self.classify_neurons(peak_heights, sparse_connections)
        
        return sparse_connections, neuron_types, peak_heights
    
    def classify_neurons(self, peak_heights: torch.Tensor, 
                        connections: torch.Tensor) -> torch.Tensor:
        """Classify neurons based on their output effects"""
        n_neurons = peak_heights.shape[0]
        
        # For each neuron, look at its outgoing connections
        excitatory_score = torch.zeros(n_neurons, device=peak_heights.device)
        
        for i in range(n_neurons):
            outgoing = connections[i, :]
            if outgoing.sum() > 0:
                # Average effect on downstream neurons
                effects = peak_heights[i, outgoing]
                excitatory_score[i] = effects.mean()
        
        # Classify based on score
        neuron_types = (excitatory_score > 0).int()  # 1 for excitatory, 0 for inhibitory
        
        return neuron_types

class NeuralConnectivityPipeline:
    """Main pipeline for neural connectivity analysis"""
    def __init__(self, hdf5_path: str, device: str = 'cuda', use_fft: bool = True, clip_bins: int = None, max_lag: int = 20):
        self.device = device
        self.data_loader = SpikeDataLoader(hdf5_path, device, clip_bins=clip_bins, max_lag=max_lag)
        self.use_fft = use_fft
        
    def run_analysis(self, max_lag: int = 20, 
                     n_shuffles: int = 100, 
                     sparsity: float = 0.15,
                     alpha: float = 0.01,
                     causal_win_start: int = 5,
                     causal_win_end: int = 20) -> Dict[str, np.ndarray]:
        """Run complete connectivity analysis pipeline"""
        
        # Step 1: Compute cross-correlograms
        print("Computing cross-correlograms on GPU...")
        start_time = time.time()
        
        if self.use_fft:
            print("Using FFT-based method for cross-correlograms.")
            xcorr_computer = FastCrossCorrelogramGPU(max_lag=max_lag)
            xcorr_full = xcorr_computer.compute_all_pairs_fft(self.data_loader)
            xcorr_full = torch.tensor(xcorr_full, device=self.device)
        else:
            print("Using direct GPU method for cross-correlograms.")
            xcorr_computer = CrossCorrelogramGPU(max_lag=max_lag)
            xcorr_full, _ = xcorr_computer.compute_all_pairs(self.data_loader)
        
        print(f"Cross-correlogram computation took {time.time() - start_time:.2f} seconds")
        
        # Step 2: Statistical significance
        print("Testing statistical significance...")
        start_time = time.time()
        
        tester = SignificanceTester(n_shuffles=n_shuffles, use_fft=self.use_fft, alpha=alpha)
        p_values, thresholds = tester.shuffle_test_gpu(
            self.data_loader.spike_times, 
            xcorr_full, 
            self.data_loader.n_bins,
            causal_win_start=causal_win_start,
            causal_win_end=causal_win_end
        )
        
        print(f"Significance testing took {time.time() - start_time:.2f} seconds")
        
        # Step 3: Network inference
        print("Inferring network structure...")
        start_time = time.time()
        
        lag_bins = torch.arange(-max_lag, max_lag+1, device=self.device)
        inferencer = NetworkInference(sparsity=sparsity)
        connections, neuron_types, weights = inferencer.infer_connectivity(
            xcorr_full, p_values, lag_bins, alpha=alpha,
            causal_win_start=causal_win_start, causal_win_end=causal_win_end
        )
        
        print(f"Network inference took {time.time() - start_time:.2f} seconds")
        
        # Step 4: Extract excitatory network
        excitatory_mask = neuron_types == 1
        excitatory_network = connections.clone()
        excitatory_network[~excitatory_mask, :] = 0  # Remove inhibitory rows
        
        # Convert to numpy for output
        results = {
            'adjacency_matrix': connections.cpu().numpy(),
            'excitatory_network': excitatory_network.cpu().numpy(),
            'neuron_types': neuron_types.cpu().numpy(),  # 1 for excitatory, 0 for inhibitory
            'weights': weights.cpu().numpy(),
            'p_values': p_values.cpu().numpy(),
            'cross_correlograms': xcorr_full.cpu().numpy()
        }

        # Step 5: Compare with ground truth if available
        if self.data_loader.true_network_matrix is not None:
            results['true_network_matrix'] = self.data_loader.true_network_matrix
            comparison_metrics = compare_with_ground_truth(
                results['adjacency_matrix'],
                results['neuron_types'],
                results['true_network_matrix']
            )
            # Add metrics to results dict for saving
            for key, value in comparison_metrics.items():
                results[f'gt_comp_{key}'] = np.array(value)

        # Print summary statistics
        n_neurons = self.data_loader.n_neurons
        n_excitatory = excitatory_mask.sum().item()
        n_inhibitory = n_neurons - n_excitatory
        n_connections = connections.sum().item()
        
        # Avoid division by zero if there's only one neuron
        if n_neurons > 1:
            actual_sparsity = n_connections / (n_neurons * (n_neurons - 1))
        else:
            actual_sparsity = 0.0

        print(f"\nAnalysis complete!")
        print(f"Neurons: {n_neurons} ({n_excitatory} excitatory, {n_inhibitory} inhibitory)")
        print(f"Connections found: {n_connections} (sparsity: {actual_sparsity:.3f})")
        
        return results

def save_results(results: Dict[str, np.ndarray], output_path: str):
    """Save results to HDF5 file"""
    with h5py.File(output_path, 'w') as f:
        for key, value in results.items():
            f.create_dataset(key, data=value)
    print(f"Results saved to {output_path}")

def plot_results(results: Dict[str, np.ndarray], output_hdf5_path: str):
    """
    Plots ground truth vs. inferred connectivity and saves to a PNG file.
    """
    if 'true_network_matrix' not in results:
        print("\nSkipping results plotting: ground truth matrix not available.")
        return

    ground_truth = results['true_network_matrix']
    inferred_weights = results['weights']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    fig.suptitle('Connectivity Comparison', fontsize=16)

    # Plot 1: Ground Truth
    gt_max_abs = np.max(np.abs(ground_truth))
    if gt_max_abs == 0: gt_max_abs = 1
    im1 = ax1.imshow(ground_truth, cmap='coolwarm', interpolation='none', vmin=-gt_max_abs, vmax=gt_max_abs)
    ax1.set_title('Ground Truth Network')
    ax1.set_xlabel('Target Neuron')
    ax1.set_ylabel('Source Neuron')
    fig.colorbar(im1, ax=ax1, label="True Connection Strength")

    # Plot 2: Inferred Weights
    inf_max_abs = np.max(np.abs(inferred_weights))
    if inf_max_abs == 0: inf_max_abs = 1
    
    im2 = ax2.imshow(inferred_weights, cmap='coolwarm', interpolation='none', vmin=-inf_max_abs, vmax=inf_max_abs)
    ax2.set_title('Inferred Connection Weights')
    ax2.set_xlabel('Target Neuron')
    #ax2.set_ylabel('Source Neuron')
    fig.colorbar(im2, ax=ax2, label="Inferred Strength")

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle

    # Save the figure
    png_path = output_hdf5_path.replace('.h5', '.results.png')
    plt.savefig(png_path)
    print(f"\nSaved results visualization to: {png_path}")
    plt.show()
    plt.close(fig)

def compare_with_ground_truth(inferred_matrix: np.ndarray,
                              inferred_neuron_types: np.ndarray,
                              ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compares the inferred connectivity matrix with the ground truth, splitting by connection type.
    Assumes ground truth uses positive values for excitatory and negative values for inhibitory.
    """
    if ground_truth is None:
        return {}

    # --- Helper function for metrics ---
    def _calculate_metrics(inferred_conn, truth_conn):
        # Ignore diagonal (self-connections)
        np.fill_diagonal(inferred_conn, 0)
        np.fill_diagonal(truth_conn, 0)
        
        TP = np.sum(inferred_conn & truth_conn)
        FP = np.sum(inferred_conn & ~truth_conn)
        FN = np.sum(~inferred_conn & truth_conn)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'TP': float(TP), 'FP': float(FP), 'FN': float(FN),
                'precision': precision, 'recall': recall, 'f1_score': f1_score}

    all_metrics = {}
    
    # --- 1. Overall Comparison (ignoring type) ---
    print("\n--- Ground Truth Comparison (Overall) ---")
    inferred_overall = inferred_matrix.astype(bool)
    truth_overall = ground_truth != 0
    overall_metrics = _calculate_metrics(inferred_overall, truth_overall)
    for key, val in overall_metrics.items():
        all_metrics[f'overall_{key}'] = val
    
    # Create copies for counting to ignore diagonal
    inferred_overall_no_diag = inferred_overall.copy(); np.fill_diagonal(inferred_overall_no_diag, 0)
    truth_overall_no_diag = truth_overall.copy(); np.fill_diagonal(truth_overall_no_diag, 0)
    num_inferred_overall = np.sum(inferred_overall_no_diag)
    num_truth_overall = np.sum(truth_overall_no_diag)
    
    print(f"  Found {num_inferred_overall} connections vs. {num_truth_overall} in truth (ignoring diagonal).")
    print(f"  TP: {int(overall_metrics['TP'])}, FP: {int(overall_metrics['FP'])}, FN: {int(overall_metrics['FN'])}")
    print(f"  Precision: {overall_metrics['precision']:.3f}, Recall: {overall_metrics['recall']:.3f}, F1-Score: {overall_metrics['f1_score']:.3f}")

    # --- 2. Excitatory Comparison ---
    print("\n--- Ground Truth Comparison (Excitatory) ---")
    truth_excitatory = ground_truth > 0
    
    excitatory_neuron_mask = inferred_neuron_types == 1
    inferred_excitatory_mask = np.zeros_like(inferred_matrix, dtype=bool)
    inferred_excitatory_mask[excitatory_neuron_mask, :] = True
    inferred_excitatory = inferred_matrix.astype(bool) & inferred_excitatory_mask

    excitatory_metrics = _calculate_metrics(inferred_excitatory, truth_excitatory)
    
    # Create copies for counting to ignore diagonal
    inferred_excitatory_no_diag = inferred_excitatory.copy(); np.fill_diagonal(inferred_excitatory_no_diag, 0)
    truth_excitatory_no_diag = truth_excitatory.copy(); np.fill_diagonal(truth_excitatory_no_diag, 0)
    num_inferred_excitatory = np.sum(inferred_excitatory_no_diag)
    num_truth_excitatory = np.sum(truth_excitatory_no_diag)

    if num_truth_excitatory > 0:
        for key, val in excitatory_metrics.items():
            all_metrics[f'excitatory_{key}'] = val
        print(f"  Found {num_inferred_excitatory} connections vs. {num_truth_excitatory} in truth (ignoring diagonal).")
        print(f"  TP: {int(excitatory_metrics['TP'])}, FP: {int(excitatory_metrics['FP'])}, FN: {int(excitatory_metrics['FN'])}")
        print(f"  Precision: {excitatory_metrics['precision']:.3f}, Recall: {excitatory_metrics['recall']:.3f}, F1-Score: {excitatory_metrics['f1_score']:.3f}")
    else:
        print(f"  Found {num_inferred_excitatory} connections. No excitatory connections in ground truth (ignoring diagonal).")


    # --- 3. Inhibitory Comparison ---
    print("\n--- Ground Truth Comparison (Inhibitory) ---")
    truth_inhibitory = ground_truth < 0

    inhibitory_neuron_mask = inferred_neuron_types == 0
    inferred_inhibitory_mask = np.zeros_like(inferred_matrix, dtype=bool)
    inferred_inhibitory_mask[inhibitory_neuron_mask, :] = True
    inferred_inhibitory = inferred_matrix.astype(bool) & inferred_inhibitory_mask

    inhibitory_metrics = _calculate_metrics(inferred_inhibitory, truth_inhibitory)
    
    # Create copies for counting to ignore diagonal
    inferred_inhibitory_no_diag = inferred_inhibitory.copy(); np.fill_diagonal(inferred_inhibitory_no_diag, 0)
    truth_inhibitory_no_diag = truth_inhibitory.copy(); np.fill_diagonal(truth_inhibitory_no_diag, 0)
    num_inferred_inhibitory = np.sum(inferred_inhibitory_no_diag)
    num_truth_inhibitory = np.sum(truth_inhibitory_no_diag)

    if num_truth_inhibitory > 0:
        for key, val in inhibitory_metrics.items():
            all_metrics[f'inhibitory_{key}'] = val
        print(f"  Found {num_inferred_inhibitory} connections vs. {num_truth_inhibitory} in truth (ignoring diagonal).")
        print(f"  TP: {int(inhibitory_metrics['TP'])}, FP: {int(inhibitory_metrics['FP'])}, FN: {int(inhibitory_metrics['FN'])}")
        print(f"  Precision: {inhibitory_metrics['precision']:.3f}, Recall: {inhibitory_metrics['recall']:.3f}, F1-Score: {inhibitory_metrics['f1_score']:.3f}")
    else:
        print(f"  Found {num_inferred_inhibitory} connections. No inhibitory connections in ground truth (ignoring diagonal).")

    print("------------------------------------------")
    return all_metrics

def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Neural Connectivity Analysis Pipeline using Cross-Correlograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, default='/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp3/input_fitter/',
                        help="Path to the data directory.")
    parser.add_argument('--file_name', type=str, default='daleM40-a03d69a-a8a35f7.spikes.h5',
                        help='Name of the input HDF5 file.')
    parser.add_argument('-o', '--output_file', type=str, default='connectivity_results.h5', 
                        help='Path to output HDF5 results file.')
    
    # Analysis parameters
    parser.add_argument('--max_lag', type=int, default=20, 
                        help='Maximum lag for cross-correlogram (in bins).')
    parser.add_argument('--n_shuffles', type=int, default=30, 
                        help='Number of shuffles for significance testing.')
    parser.add_argument('--sparsity', type=float, default=0.15, 
                        help='Target sparsity for network inference.')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Significance level (p-value threshold) for connection filtering.')
    parser.add_argument('--causal_win_start', type=int, default=1,
                        help='Start of the causal window for peak detection (in bins/ms).')
    parser.add_argument('--causal_win_end', type=int, default=5,
                        help='End of the causal window for peak detection (in bins/ms).')
    
    # Execution options
    parser.add_argument('--clip_bins', type=int, default=None,
                        help='Clip number of time bins to this value.')
    parser.add_argument('--no_fft', action='store_true', 
                        help='Disable FFT-based cross-correlogram computation (slower).')
    parser.add_argument('--device', type=str, default=None, 
                        help="Device to use ('cuda' or 'cpu'). Auto-detects if not specified.")
    
    args = parser.parse_args()
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    # --- Construct full input path ---
    full_input_path = os.path.join(args.data_path, args.file_name)

    # --- Pre-computation checks ---
    if not args.no_fft and not CUPY_AVAILABLE:
        print("\nError: --no_fft flag was not specified, but CuPy is not available.")
        print("Please install CuPy or run with the --no_fft flag to use the direct method.")
        exit(1)

    if args.max_lag < args.causal_win_end:
        print(f"\nError: --max_lag ({args.max_lag}) must be greater than or equal to --causal_win_end ({args.causal_win_end}).")
        print("The 'max_lag' defines the total window for calculation, while 'causal_win_end' defines the specific area to search within it.")
        exit(1)

    # --- Set up device ---
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: --device='cuda' was specified but CUDA is not available. Falling back to CPU.")
        device = 'cpu'

    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU")
        
    # --- Run analysis ---
    try:
        pipeline = NeuralConnectivityPipeline(
            full_input_path, 
            device=device, 
            use_fft=not args.no_fft, 
            clip_bins=args.clip_bins,
            max_lag=args.max_lag
        )
        results = pipeline.run_analysis(
            max_lag=args.max_lag,
            n_shuffles=args.n_shuffles,
            sparsity=args.sparsity,
            alpha=args.alpha,
            causal_win_start=args.causal_win_start,
            causal_win_end=args.causal_win_end
        )
        
        # --- Save results ---
        save_results(results, args.output_file)
        return results, args
        
    except FileNotFoundError:
        print(f"\nError: Could not find input file '{full_input_path}'")
        print("Please ensure the HDF5 file exists with spike data in format:")
        print("  - Dataset 'spikes_data' with shape (n_neurons, n_time_bins)")
        print("  - Boolean dtype where True indicates a spike")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    
    return None, None

# Example usage
if __name__ == "__main__":
    results, args = main()
    if results:
        plot_results(results, args.output_file)
