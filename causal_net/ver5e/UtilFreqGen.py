import numpy as np
import matplotlib.pyplot as plt
import argparse

def gen_exponential_freq(num_samples, scale1, scale2, mix_ratio, min_freq, max_freq):
    """
    Generates frequency data by combining two exponential distributions with a cutoff.

    This function uses rejection sampling: it generates values and discards any
    that are above the specified max_freq until the desired number of samples
    has been collected.

    Args:
        num_samples (int): The total number of data points to generate.
        scale1 (float): The scale parameter for the first (primary) distribution.
        scale2 (float): The scale parameter for the second distribution.
        mix_ratio (float): The proportion of samples from the first distribution.
        max_freq (float): The maximum allowed frequency. Values above this are rejected.

    Returns:
        np.ndarray: A 1D array of the final frequency values.
    """
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError("mix_ratio must be between 0.0 and 1.0")

    # Use a list to collect valid samples
    frequencies = []
    
    # Loop until we have collected the required number of samples
    while len(frequencies) < num_samples:
        # Decide which distribution to draw from based on the mix_ratio
        if np.random.rand() < mix_ratio:
            sample = np.random.exponential(scale=scale1)
        else:
            sample = np.random.exponential(scale=scale2)
        
        # Rejection step: only keep the sample if it's within the valid range
        if sample < min_freq: continue
        if sample > max_freq: continue
        frequencies.append(sample)
    freq=np.array(frequencies)
    # Shuffle the array in-place before returning
    np.random.shuffle(freq)
    return freq


def plot_freq(freq_data, title="Input Channel"):
    """
    Creates a histogram plot from frequency data, styled like the example.

    This function is compatible with older versions of Python (pre-3.6)
    by using the .format() string method instead of f-strings.

    Args:
        freq_data (np.ndarray): The 1D array of frequency data to plot.
        title (str): The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(freq_data, bins=20, log=True)

    median_val = np.median(freq_data)
    ax.axvline(median_val, color='r', linestyle='--', linewidth=1.5)

    y_max = ax.get_ylim()[1]
    median_text = f"median: {median_val:.2f}, N={freq_data.shape[0]}"
    ax.text( x=median_val * 1.1,  y=y_max * 0.7, s=median_text,  color='red')

    ax.set_xlabel("avr frequency (Hz)")
    ax.set_ylabel("num channels")
    ax.set_title(title)
    ax.grid(True)
    plt.show()

def main():
    """
    Main function to parse arguments, generate data, and plot the distribution.
    """
    parser = argparse.ArgumentParser(
        description="Generate and plot a frequency distribution from a mix of two exponential distributions."
    )
    parser.add_argument(
        '-n', '--num_samples', type=int, default=300,
        help='Total number of samples (channels) to generate.'
    )
    parser.add_argument(
        '--scale1', type=float, default=3,
        help='Scale for the primary (low frequency) exponential distribution.'
    )
    parser.add_argument(
        '--scale2', type=float, default=0.4,
        help='Scale for the secondary (higher frequency) exponential distribution.'
    )
    parser.add_argument(
        '-m', '--mix_ratio', type=float, default=0.7,
        help='Mix ratio (0.0 to 1.0) for the primary distribution.'
    )
    parser.add_argument(
        '--maxFreq', type=float, default=15.0,
        help='Maximum frequency value. Generated values above this will be discarded.'
    )
    parser.add_argument(
        '--minFreq', type=float, default=1.0,
        help='Minimal frequency value. Generated values above this will be discarded.'
    )
    
    args = parser.parse_args()

    print("--- Generator Parameters ---")
    print("Total Samples: {}".format(args.num_samples))
    print("Primary Scale: {}".format(args.scale1))
    print("Secondary Scale: {}".format(args.scale2))
    print("Mix Ratio: {}".format(args.mix_ratio))
    print("Min Frequency: {}".format(args.minFreq))
    print("Max Frequency: {}".format(args.maxFreq))
    print("----------------------------")

    frequencies = gen_exponential_freq(
        num_samples=args.num_samples,
        scale1=args.scale1,
        scale2=args.scale2,
        mix_ratio=args.mix_ratio,
        min_freq=args.minFreq,
        max_freq=args.maxFreq
    )
    
    plot_freq(frequencies, title="Input Channel")
    print("Done.")

if __name__ == "__main__":
    main()
