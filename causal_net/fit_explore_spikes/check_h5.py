import h5py
import numpy as np
import sys

def check_h5_file(filepath):
    """
    A simple diagnostic script to inspect the contents of a 'true_network_matrix'
    dataset within an HDF5 file.
    """
    print(f"--- Running Diagnostic Check on: {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            if 'true_network_matrix' not in f:
                print("Error: Dataset 'true_network_matrix' not found in the file.")
                return

            print("Successfully opened file and found 'true_network_matrix'.")
            
            gt_matrix = f['true_network_matrix'][:]
            
            print("\n--- Matrix Properties ---")
            print(f"  Shape: {gt_matrix.shape}")
            print(f"  Data type: {gt_matrix.dtype}")
            print(f"  Min value: {gt_matrix.min()}")
            print(f"  Max value: {gt_matrix.max()}")
            print(f"  Mean value: {gt_matrix.mean():.4f}")

            unique_vals, counts = np.unique(gt_matrix, return_counts=True)
            
            print("\n--- Content Analysis ---")
            print(f"  Total non-zero elements: {np.count_nonzero(gt_matrix)}")
            print(f"  Number of '1's (Excitatory): {np.sum(gt_matrix == 1)}")
            print(f"  Number of '-1's (Inhibitory): {np.sum(gt_matrix == -1)}")
            
            print("\n  Full breakdown of unique values found:")
            for val, count in zip(unique_vals, counts):
                print(f"    - Value '{val}': found {count} times")
            
            print("\n--- Conclusion ---")
            print("This script provides an independent check of the HDF5 file's contents.")
            print("Please compare these numbers with the diagnostic output from the main script.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_h5.py <path_to_your_hdf5_file>")
    else:
        check_h5_file(sys.argv[1]) 