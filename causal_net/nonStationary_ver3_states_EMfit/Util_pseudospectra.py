#!/usr/bin/env python3
import numpy as np
import time

def create_example_matrices(size=100):
    A_base = np.diag(np.linspace(-5, -0.5, size)) + np.diag(np.ones(size - 2) * 4, k=2)
    sparsity = 0.15
    random_connections = np.random.randn(size, size)
    mask = np.random.rand(size, size) > sparsity
    random_connections[mask] = 0
    A_true = A_base + random_connections * 0.1

    noise = np.random.randn(size, size) * 0.10
    noise_mask = np.random.rand(size, size) > sparsity
    noise[noise_mask] = 0
    A_fit = A_true + noise

    return A_true, A_fit

def compute_pseudospectrum(A, npts, minY):
    print(f'computing pseudospectrum  A:{A.shape} .... ')
    eigs = np.linalg.eigvals(A)

    # Calculate bounds
    real_min, real_max = np.real(eigs).min(), np.real(eigs).max()
    imag_min, imag_max = np.imag(eigs).min(), np.imag(eigs).max()
    
    real_pad = (real_max - real_min) * 0.2
    imag_pad = (imag_max - imag_min) * 0.2

    bbox = [
        real_min - real_pad, 
        min(real_max + real_pad, 1.0),
        imag_min,
        imag_max + imag_pad
    ]
    
    x_coords = np.linspace(bbox[0], bbox[1], npts)
    y_coords = np.linspace(minY, bbox[3], npts)
    #y_coords = np.linspace(bbox[2], bbox[3], npts)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = X + 1j * Y
    
    sigma_grid = np.zeros_like(Z, dtype=float)
    I = np.eye(A.shape[0])

    for i in range(npts):
        for j in range(npts):
            z = Z[i, j]
            s = np.linalg.svd(z * I - A, compute_uv=False)
            sigma_grid[i, j] = s[-1]
    
    return X, Y, sigma_grid, eigs

def plot_pseudospectra(X, Y, sigma_grid, eigs, minY, title, ax, epsMin):
    levels = np.logspace(-2.5, -0.5, 10)  # Use 10 contour levels

    # Plot the normal contour lines
    contour = ax.contour(X, Y, sigma_grid, levels=levels, cmap='viridis',  linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt='ε=%.3f')

    # Create a mask for the area greater than epsMin
    mask = sigma_grid > epsMin
    sigma_grid_masked = np.ma.array(sigma_grid, mask=mask)  # Masking areas greater than epsMin

    # Fill the area below epsMin
    contour_fill = ax.contourf(X, Y, sigma_grid_masked, levels=np.linspace(0, epsMin, 10), 
                                colors=['lightgreen'], alpha=0.5)
    
    # Scatter plot for eigenvalues
    ax.scatter(np.real(eigs), np.imag(eigs), color='red', s=15, zorder=3, label='Eigenvalues')
    
    # Vertical line at Re=0
    ax.axvline(0, color='black', linestyle='--', lw=1.5)
    ax.axhline(0, color='black', linestyle='--', lw=1.5)
    
    # Set titles and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    
    # Additional formatting
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    ax.set_ylim(bottom=minY)  # Clip at minY


def main(size=100, minY=-0.9, epsMin=0.032):
    A_true, A_fit = create_example_matrices(size)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    matrices_to_plot = [
        ('True Matrix ($A_{true}$)', A_true),
        ('Distorted Matrix ($A_{fit}$) ', A_fit)
    ]

    print("Starting pseudospectrum calculations...", minY)
    start_time = time.time()

    for i, (title, A) in enumerate(matrices_to_plot):
        ax = axes[i]
        print("Processing '{}'...".format(title))

        npts = 80  # Grid resolution
        X, Y, sigma_grid, eigs = compute_pseudospectrum(A, npts, minY)
        
        plot_pseudospectra(X, Y, sigma_grid, eigs, minY, title, ax, epsMin)  # Pass epsMin

    total_time = time.time() - start_time
    print(f"Calculation finished in {total_time:.2f} seconds.")

    fig.suptitle('Pseudospectrum Comparison (Clipped Imaginary Part)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    outFile = 'out/pseudospectra_comparison.png'    
    plt.savefig(outFile)
    print(f"Plot saved as '{outFile}'")
    plt.close(fig)

# --- Main execution ---
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
 
    main(size=50, minY=-0.5, epsMin=0.024)
