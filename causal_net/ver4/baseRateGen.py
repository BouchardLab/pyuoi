import numpy as np
import matplotlib.pyplot as plt

def generate_mixture_spike_frequencies(num_samples=300, 
                                       mu1=-1.1, sigma1=0.8, weight1=0.6, # steep
                                       mu2=1.0, sigma2=1.2, weight2=0.4, # wide w/ bump
                                       random_seed=42):
    """
    Generate a mixture of two log-normal distributed arrays of spike frequencies.
    """
    assert abs(weight1 + weight2 - 1.0) < 1e-6, "Weights must sum to 1"
    if random_seed is not None:
        np.random.seed(random_seed)
    n1 = int(num_samples * weight1)
    n2 = num_samples - n1
    data1 = np.random.lognormal(mean=mu1, sigma=sigma1, size=n1)
    data2 = np.random.lognormal(mean=mu2, sigma=sigma2, size=n2)
    data = np.concatenate([data1, data2])
    data = np.clip(data, 0.8, 15.)
    np.random.shuffle(data)
    return data

def plot_spike_histogram(data, ax, hiFreq=14, png_name=None, title="Synthetic mixture log-normal distribution"):
    """
    Plot the histogram of spike frequencies on a given axis and optionally save as PNG.
    """
    if hiFreq is not None:
        data = data[data <= hiFreq]
        ax.set_xlim(0, hiFreq)
    ax.hist(data, bins=25, color='salmon', alpha=0.7, log=True)
    ax.set_xlabel('spike frequency (Hz)')
    ax.set_ylabel('num features')
    ax.set_title(title)
    
    ax.grid()
    # Add stats as text
    mean = np.mean(data)
    median = np.median(data)
    p30 = np.percentile(data, 30)
    p70 = np.percentile(data, 70)
    stats = (f"Mean: {mean:.3f}\n"
             f"Median: {median:.3f}\n"
             f"30th: {p30:.3f}\n"
             f"70th: {p70:.3f}")
    ax.text(0.98, 0.98, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    if png_name is not None:
        plt.tight_layout()
        plt.savefig(png_name, dpi=150)
        print(f"Saved: {png_name}")

def main():
    # This would be your original data, replace with real data if available
    # For demonstration, let's make a "fake original" using a single log-normal
    np.random.seed(0)
  
    data1 = generate_mixture_spike_frequencies(num_samples=100)
        
    # Generate synthetic mixture data
    mix_data = generate_mixture_spike_frequencies(
        num_samples=1000,
        mu1=-1.1, sigma1=0.8, weight1=0.6,  # steep
        mu2=1.0, sigma2=1.2, weight2=0.4  # wide w/ bump
    )
    
    # Plot both for comparison
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    plot_spike_histogram(data1, axs[1], title="low stats ")
    plot_spike_histogram(mix_data, axs[0], hiFreq=14, title="Mixture log-normal (synthetic)", png_name="out/mixture_spike_histogram.png")
    plt.show()  # Show after saving

if __name__ == "__main__":
    main()
