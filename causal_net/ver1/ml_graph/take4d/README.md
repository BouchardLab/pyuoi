# Data Generation and Model Fitting Workflow for a Dynamic Neural System

This directory contains a Python-based workflow for simulating a dynamic neural system and then fitting a model to learn its underlying parameters. The system adheres to Dale's Principle, where neurons are either strictly excitatory or strictly inhibitory.

## 1. Data Generation (`gen_dale_data.py`)

This script simulates and generates training data for the neural system.

### Key Operations:
-   **`generate_dale_matrix`**: Creates a sparse weight matrix `W` for a specified number of neurons. The first `num_excite` rows are excitatory (positive off-diagonal weights), and the remaining are inhibitory (negative off-diagonal weights). Diagonal elements are always negative. The matrix is rescaled if needed to ensure system stability.
-   **`simulate_evolution`**: Simulates the system's state over time using the equation `x_{t+1} = x_t + (1/tau) * (-x_t + W @ x_t) + noise`.
-   **Outputs**:
    -   A `.npz` file containing the ground-truth `W`, connectivity mask `E`, `tau`, the full `trajectory`, and `w_dims` ([num_neuron, num_excite, num_inhibit]).
    -   A `.png` file visualizing the true `W` matrix, sample trajectories, and histograms of the different weight types.

### Usage Example:
```bash
./gen_dale_data.py --numNeuron 40 --numExcite 20 --sparse 0.15 --simName my_dale_sim
```

---

## 2. Model Fitting (`fit_dale_model.py`)

This script learns the system parameters from the generated data using a model that structurally enforces Dale's Principle.

### Key Operations:
-   **Model Architecture (V -> W)**: The model does not learn `W` directly. Instead, it learns a precursor matrix `V`. In every forward pass, `W` is dynamically computed from `V` using a fixed transformation that guarantees positive excitatory weights, negative inhibitory weights, and negative diagonals. This hard-codes the physical constraints into the model.
-   **Training**: It uses the Adam optimizer and Mean Squared Error (MSE) loss to train the `V` matrix. The training loop includes learning rate scheduling and early stopping for efficiency.
-   **Evaluation**: After training, it produces a detailed 2x4 plot showing:
    -   Training loss over time.
    -   A heatmap of the final fitted `W` matrix.
    -   Correlation and residual plots for the three distinct weight groups: diagonal, excitatory, and inhibitory.

### Usage Example:
The `gen_dale_data.py` script will print a suggested command to run the fitter.
```bash
./fit_dale_model.py --input my_dale_sim --epochs 100 --lr 0.01
``` 