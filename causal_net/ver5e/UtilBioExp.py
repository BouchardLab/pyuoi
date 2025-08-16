import numpy as np
from scipy.ndimage import label

def create_clusters_mask(X, th):
    """
    Creates a boolean mask for a 1D NumPy array.

    A contiguous cluster of non-zero values is marked as True if any value
    within that cluster is greater than or equal to the threshold 'th'.
    Clusters are assumed to be separated by zero values.

    Parameters:
    X (np.ndarray): Input 1D NumPy array.
    th (float): The threshold value.

    Returns:
    np.ndarray: A boolean mask with the same shape as X.
    """
    # Step 1: Find all contiguous clusters of non-zero values.
    # The `label` function assigns a unique integer to each cluster.
    # e.g., [1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3]
    labeled_array, num_clusters = label(X > 0)
    
    # Step 2: Initialize the final mask to all False.
    final_mask = np.zeros_like(X, dtype=bool)
    
    # Step 3: Iterate through each cluster found by the label function.
    # We start from 1 because 0 is the background (the zeros in X).
    for i in range(1, num_clusters + 1):
        # Create a boolean mask for the current cluster only.
        current_cluster_mask = (labeled_array == i)
        
        # Step 4: Check if ANY value within this specific cluster meets the threshold.
        if np.any(X[current_cluster_mask] >= th):
            # Step 5: If the condition is met, mark this entire cluster as True in our final mask.
            final_mask[current_cluster_mask] = True
            
    return final_mask

if __name__=="__main__":

    # Example usage
    X = np.array([1, 2, 3, 0, 2, 6, 2, 0, 1, 2, 8, 0, 0])
    threshold = 5
    mask = create_clusters_mask(X, threshold)
    
    # Print the output in two columns
    print(f"Input  | Mask , thr:{threshold}")
    print("-" * 25)
    for x_val, mask_val in zip(X, mask):
        print(f"{x_val:<4} | {mask_val}")
