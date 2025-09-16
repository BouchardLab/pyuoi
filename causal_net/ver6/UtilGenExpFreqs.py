import numpy as np
import matplotlib.pyplot as plt

def gen_realistic_freqs(min_freq, max_freq, trapezoid_height, num_samples, sigma):
    """
    Generate samples using a combination of Gaussian and trapezoidal distributions.
    
    Parameters:
    - min_freq: Minimum frequency value.
    - max_freq: Maximum frequency value.
    - trapezoid_height: Height of the trapezoidal distribution at min_freq.
    - num_samples: Total number of samples to generate.
    - sigma: Standard deviation of the Gaussian distribution.
    
    Returns:
    - samples: Array of generated samples.
    """
    
    # Define the Gaussian PDF
    gaussian_pdf = lambda f: np.exp(- (f / sigma) ** 2 / 2)

    # Define the trapezoidal PDF using vectorized operations
    trapezoidal_pdf = lambda f: np.where(
        (f >= min_freq) & (f <= max_freq),
        trapezoid_height * (1 - (f - min_freq) / (max_freq - min_freq) * 0.9),
        0
    )

    # Create a combined PDF
    combined_pdf = lambda f: gaussian_pdf(f) + trapezoidal_pdf(f)

    # Normalize the combined PDF
    x = np.linspace(min_freq, max_freq, 100)
    pdf_values = combined_pdf(x)
    normalization_factor = np.trapezoid(pdf_values, x)  # Calculate the area under the curve
    normalized_pdf = pdf_values / normalization_factor  # Normalize the PDF

    # Draw from the normalized PDF
    samples = np.random.choice(x, size=num_samples, p=normalized_pdf/np.sum(normalized_pdf))

    return samples

if __name__ == '__main__':

    # Parameters
    min_freq = 1  # Set minimum frequency to 1
    max_freq = 50
    trapezoid_height = 0.15  # Height of the trapezoidal distribution
    num_samples = 150
    sigma = 3  # Standard deviation of the Gaussian

    # Generate samples
    samples = gen_realistic_freqs(min_freq, max_freq, trapezoid_height, num_samples, sigma)

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
