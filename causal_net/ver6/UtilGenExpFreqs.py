#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def gen_realistic_freqs(min_freq, max_freq, trapezoid_height, trapezoid_rmin, num_samples, sigma):
    """
    Generate samples using a combination of Gaussian and trapezoidal distributions.
    
    Parameters:
    - min_freq: Minimum frequency value.
    - max_freq: Maximum frequency value.
    - trapezoid_height: Height of the trapezoidal distribution at min_freq.
    - trapezoid_rmin: Relative reduction at max_freq (0..1).
    - num_samples: Total number of samples to generate.
    - sigma: Standard deviation of the Gaussian distribution.
    
    Returns:
    - samples: Array of generated samples.
    """
    
    # Define the Gaussian PDF (centered at 0, evaluated for f>=min_freq)
    def gaussian_pdf(f):
        return np.exp(- (f / sigma) ** 2 / 2)

    # Define the trapezoidal PDF using vectorized operations
    def trapezoidal_pdf(f):
        inside = (f >= min_freq) & (f <= max_freq)
        val = np.zeros_like(f, dtype=float)
        # Linearly decreasing from trapezoid_height at min_freq
        val[inside] = trapezoid_height * (1 - (f[inside] - min_freq) / (max_freq - min_freq) * trapezoid_rmin)
        return val

    # Create a combined PDF
    def combined_pdf(f):
        return gaussian_pdf(f) + trapezoidal_pdf(f)

    # Rejection sampling over continuous interval [min_freq, max_freq] to avoid quantization
    x_dense = np.linspace(min_freq, max_freq, 5000)
    pdf_dense = combined_pdf(x_dense)
    M = float(np.max(pdf_dense)) if np.all(np.isfinite(pdf_dense)) else 1.0
    if M <= 0:
        # Fallback: uniform if pathological configuration
        return np.random.uniform(min_freq, max_freq, size=num_samples)
    
    samples = np.empty(num_samples, dtype=float)
    filled = 0
    # Choose a reasonable batch size for efficiency
    batch_size = max(1000, num_samples * 2)
    while filled < num_samples:
        f_try = np.random.uniform(min_freq, max_freq, size=batch_size)
        u = np.random.uniform(0.0, M, size=batch_size)
        accept_mask = u < combined_pdf(f_try)
        accepted = f_try[accept_mask]
        take = min(accepted.size, num_samples - filled)
        if take > 0:
            samples[filled:filled+take] = accepted[:take]
            filled += take

    return samples

if __name__ == '__main__':

    # Parameters
    min_freq = 0.5  # Set minimum frequency 
    max_freq = 90 #(Hz)
    trapezoid_height = 0.15  # Height of the trapezoidal distribution
    trapezoid_rmin=0.3 # fraction of trapzoid at the max freq
    sigma = 5  # (Hz) Standard deviation of the Gaussian

    num_samples = 150

    # Generate samples
    samples = gen_realistic_freqs(min_freq, max_freq, trapezoid_height, trapezoid_rmin,num_samples, sigma)

    print('samples:',samples[samples<5])
    # Calculate the median of the samples
    median_value = np.median(samples)

    # Plotting the generated samples
    fig, ax = plt.subplots(figsize=(8, 3))

    # Create the histogram without normalization
    ax.hist(samples, bins=np.arange(min_freq, max_freq + 2, 2), color='steelblue', edgecolor='black', alpha=0.7)

    # Draw the median line
    ax.axvline(median_value, color='red', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')

    # Formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Combined Distribution of Gaussian and Trapezoidal', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Print parameters inside the canvas
    params_text = (f"Min Frequency: {min_freq} Hz\n"
                   f"Max Frequency: {max_freq} Hz\n"
                   f"Trapezoid Height: {trapezoid_height}\n"
                   f"Number of Samples: {num_samples}\n"
                   f"Gaussian Sigma: {sigma:.2f}\n"
                   f"Median: {median_value:.2f}")

    # Adding text to the plot
    ax.text(0.05, 0.95, params_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()
