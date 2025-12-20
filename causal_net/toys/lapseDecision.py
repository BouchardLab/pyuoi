import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_function(gammaL, gammaR, b, w, ax):
    """
    Draws the function P(yt = Right | xt) = γR + (1 - γR - γL) * (1 / (1 + e^-(wxt+b))).

    Args:
        gammaL (float): The lower bound parameter (γL).
        gammaR (float): The upper bound parameter (γR).
        b (float): The bias parameter in the logistic term.
        w (float): The weight/slope parameter in the logistic term.
        ax (matplotlib.axes.Axes): The axes (canvas) to draw on.
    """
    # Define the fixed x-range from -10 to 10 with a smooth curve
    x = np.linspace(-10, 10, 400)

    # Calculate the logistic term (sigmoid function)
    logistic_term = 1 / (1 + np.exp(-(w * x + b)))

    # Calculate the full function value
    P_right = gammaR + (1 - gammaR - gammaL) * logistic_term

    # Plot the function on the provided axes
    ax.plot(x, P_right, label=f'low: γR={gammaR}, high: γL={gammaL}, w={w}, b={b}')
    ax.set_xlabel('$x_t$')
    ax.set_ylabel('$P(y_t = Right | x_t)$')
    ax.set_title('Classic lapse model for sensory decision-making')
    ax.legend()
    ax.grid(True)
    ax.set_xlim([-10, 10]) # Ensure the fixed x-range is applied
    ax.set_ylim()   # Probabilities are typically between 0 and 1

def main():
    """
    Parses command line arguments and generates the plot.
    """
    parser = argparse.ArgumentParser(description="Plot a modified logistic function with specified parameters.")
    
    # Add arguments for gammaL, gammaR, b, and w
    parser.add_argument('--gammaL', type=float, default=0.2, help="Value for the lower bound parameter gammaL.")
    parser.add_argument('--gammaR', type=float, default=0.7, help="Value for the upper bound parameter gammaR.")
    parser.add_argument('--b', type=float, default=0.0, help="Value for the bias parameter b (default: 0.0).")
    parser.add_argument('--w', type=float, default=1.0, help="Value for the weight/slope parameter w (default: 1.0).")
    
    args = parser.parse_args()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    # Call the plotting function with user-provided arguments
    plot_function(args.gammaL, args.gammaR, args.b, args.w, ax)

    # Define the filename and save the figure
    filename = 'out/function_plot.png'
    fig.savefig(filename)
    print(f"Plot saved successfully as '{os.path.abspath(filename)}'")

if __name__ == "__main__":
    main()

