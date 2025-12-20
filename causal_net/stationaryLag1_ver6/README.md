# Neural Connectivity Inference with LASSO Poisson GLM

This package provides a complete pipeline for inferring neural connectivity from spike count data using LASSO-regularized Poisson Generalized Linear Models (GLM) with statistical significance testing via False Discovery Rate (FDR) control.

## Overview

The pipeline implements a state-of-the-art approach for neural connectivity analysis that combines:
- **Poisson GLM modeling** for biologically realistic spike count data
- **LASSO regularization** for sparse connectivity estimation
- **Bootstrap resampling** for statistical robustness
- **FDR-controlled edge selection** for multiple comparison correction
- **Multi-GPU distributed training** for computational efficiency

### Key Features
- Support for both simulated and experimental neural data
- Distributed training across multiple GPUs using PyTorch DDP
- Bootstrap-based statistical inference with configurable sample sizes
- Row-wise FDR control for conservative edge selection
- Comprehensive visualization and evaluation tools
- Dale's principle simulation for method validation

## Workflow Pipeline

```
Data Preparation → Bootstrap Training → Edge Selection → Evaluation
      ↓                    ↓                ↓            ↓
[prep_bioexp.py]    [bootsFit.sh]    [selectEdges_FDR.py]  [eval_fitLasso.py]
[sim_dalePoisson.py]      ↓                             [view_bioexp.py]
                   [fitLasso4GPU.sh]
                         ↓
                [fit_lassoPoisson.py]
```

## Installation & Requirements

- Python 3.8+
- PyTorch with CUDA support
- NumPy, SciPy, Matplotlib
- statsmodels (for FDR correction)
- Custom toolbox modules (PlotterBackbone, Util_NumpyIO)

## Executables Reference

### Data Preparation Scripts

#### `prep_bioexp.py`
**Purpose**: Preprocessing pipeline for experimental neural data from Roy/Mandar laboratory  
**Input**: Raw experimental neural recordings  
**Output**: Standardized `.spikes.npz` files with spike count matrices and metadata  
**Usage**: 
```bash
./prep_bioexp.py --sessionName B6J_250619_M08020_000093_Well000 --inputPath /path/to/raw/data/
```

#### `sim_dalePoisson.py`
**Purpose**: Simulates Dale's principle neural networks with Poisson spiking dynamics  
**Input**: Network parameters (num_neurons, connectivity strength, etc.)  
**Output**: `.spikes.npz` and `.simTruth.npz` files with simulated data and ground truth  
**Usage**:
```bash
./sim_dalePoisson.py --num_neurons 50 --num_excite 35 --num_steps 10000 --dataName test_dale
```

#### `view_bioexp.py`
**Purpose**: Visualization tool for biological experiment data quality assessment  
**Input**: `.spikes.npz` files  
**Output**: Interactive plots for exploratory data analysis  
**Usage**:
```bash
./view_bioexp.py --dataPath /path/to/data/ --sessionName mydata -p a
```

### Model Training Scripts

#### `bootsFit.sh`
**Purpose**: Bootstrap wrapper for running multiple LASSO training iterations  
**Function**: Orchestrates multiple training runs with different random seeds/data splits  
**Key Features**:
- Configurable number of bootstrap iterations (--num_bootstraps)
- Support for time-shuffled control data (--shuffleTime)
- Automatic naming convention (dataName-boot0, dataName-boot1, etc.)
- Progress tracking and error handling
- Pass-through of training parameters to underlying scripts

**Usage**:
```bash
./bootsFit.sh --dataName mydata --num_bootstraps 10 --num_epochs 100 --dropDataFrac 0.5
```

#### `fitLasso4GPU.sh`
**Purpose**: Multi-GPU wrapper for distributed LASSO Poisson training  
**Function**: Sets up GPU environment and executes fit_lassoPoisson.py  
**Key Features**:
- Multi-GPU configuration (4 GPUs default)
- Optimized environment variables (OMP_NUM_THREADS=1, etc.)
- Default hyperparameters (batch_size=2048, lr=1e-3, L1_alpha=1e-3)
- Execution timing and logging

**Usage**: Called automatically by bootsFit.sh, not typically run directly

#### `fit_lassoPoisson.py`
**Purpose**: Core distributed training of Poisson GLM with LASSO regularization  
**Input**: `.spikes.npz` files with neural spike count data  
**Output**: `.lassoFit.npz` files with fitted connectivity matrices and metadata  
**Key Features**:
- PyTorch DistributedDataParallel for multi-GPU training
- Custom Poisson negative log-likelihood loss
- L1 regularization for sparse connectivity
- Support for time decorrelation and data shuffling
- Automatic model checkpointing

**Usage**:
```bash
# Multi-GPU (via torchrun)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./fit_lassoPoisson.py --dataName mydata --num_epochs 100

# Single GPU
./fit_lassoPoisson.py --dataName mydata --num_epochs 100
```

### Statistical Analysis Scripts

#### `selectEdges_FDR.py`
**Purpose**: Row-wise FDR edge selection from LASSO bootstrap results  
**Input**: Multiple `.lassoFit.npz` files from bootstrap training  
**Output**: `.selFdr.lassoFit.npz` with FDR-selected edges and statistics  
**Key Features**:
- Row-wise Benjamini-Hochberg FDR correction
- Configurable number of real vs shuffled bootstraps
- Empirical p-value computation using pooled null distribution
- Comprehensive statistical summaries and edge selection metrics

**Usage**:
```bash
# Single value (duplicated for real/shuffled)
./selectEdges_FDR.py --dataName mydata --num_bootstraps 10 --alpha 0.01

# Different numbers of real vs shuffled
./selectEdges_FDR.py --dataName mydata --num_bootstraps 8 12 --alpha 0.01
```

### Evaluation Scripts

#### `eval_fitLasso.py`
**Purpose**: Comprehensive evaluation and visualization of LASSO fitting results  
**Input**: `.lassoFit.npz` files (individual fits or FDR-selected)  
**Output**: Statistical summaries and visualization plots  
**Key Features**:
- Connectivity matrix visualization with frequency sorting
- Weight distribution analysis and sparsity metrics
- Network structure plots for excitatory/inhibitory connections
- Ground truth comparison for simulated data

**Usage**:
```bash
./eval_fitLasso.py --dataName mydata-selFdr --dataPath /path/to/data/ -p ab
```

#### `eval_fitRegress.py`
**Purpose**: Evaluation of regression-based connectivity analysis  
**Input**: Regression fit results  
**Output**: Alternative connectivity analysis using regression methods  
**Usage**:
```bash
./eval_fitRegress.py --dataName mydata --dataPath /path/to/data/ -p ab
```

## Utility Modules

### Core Utilities
- **`PoissonGLModel.py`**: PyTorch implementation of Poisson GLM with dual parameterization modes
- **`UtilTorch.py`**: GPU setup, data preprocessing, and distributed training utilities
- **`UtileSelectFDR.py`**: Statistical functions for FDR edge selection and bootstrap analysis
- **`UtilDalePoisson.py`**: Dale's principle simulation utilities and connectivity analysis
- **`UtilBioExp.py`**: Biological experiment data processing and cluster detection
- **`UtilFreqGen.py`**: Realistic neural firing frequency distribution generation

### Plotting Modules
- **`PlotterFitEval.py`**: LASSO fit evaluation and connectivity visualization
- **`PlotterBioExp.py`**: Biological experiment data visualization
- **`PlotterSimPoisson.py`**: Simulated neural network and Dale's principle plots

## Typical Analysis Workflow

### 1. Data Preparation
```bash
# For experimental data
./prep_bioexp.py --sessionName myexperiment --inputPath /raw/data/

# For simulated data
./sim_dalePoisson.py --num_neurons 50 --num_excite 35 --dataName mysim
```

### 2. Bootstrap Training
```bash
# Train multiple bootstrap iterations
./bootsFit.sh --dataName mydata --num_bootstraps 10 --num_epochs 200 --dropDataFrac 0.5

# Optional: Train time-shuffled controls
./bootsFit.sh --dataName mydata --num_bootstraps 10 --shuffleTime --num_epochs 200
```

### 3. Statistical Edge Selection
```bash
# Apply FDR correction with real and shuffled data
./selectEdges_FDR.py --dataName mydata --num_bootstraps 10 --alpha 0.01 -p af
```

### 4. Results Evaluation
```bash
# Visualize selected connectivity
./eval_fitLasso.py --dataName mydata-selFdr --dataPath /path/to/results/ -p abc

# Compare with ground truth (for simulations)
./eval_fitLasso.py --dataName mysim-selFdr --dataPath /path/to/results/ -p abcd
```

## Output File Conventions

- **`.spikes.npz`**: Preprocessed spike count matrices
- **`.simTruth.npz`**: Ground truth connectivity (simulations only)
- **`dataName-bootX.lassoFit.npz`**: Individual bootstrap training results
- **`dataName-shufX.lassoFit.npz`**: Time-shuffled control results
- **`dataName-selFdr.lassoFit.npz`**: FDR-selected final connectivity

## Performance Considerations

- **GPU Memory**: Ensure sufficient GPU memory for large networks (>100 neurons)
- **Bootstrap Size**: More bootstraps improve statistical power but increase computation time
- **Batch Size**: Adjust based on GPU memory and network size
- **Distributed Training**: Use multiple GPUs for networks >50 neurons or long recordings

## Citation

If you use this code, please cite the appropriate papers describing the methods implemented here.
