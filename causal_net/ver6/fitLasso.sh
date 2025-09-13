#!/bin/bash
#
# fitLasso.sh - Wrapper script for distributed Lasso Poisson training
#
# Usage: ./fitLasso.sh [OPTIONS]
#
# This script runs fit_lassoPoisson.py with optimized multi-GPU settings

# Default values
DATANAME="daleM80_285c84"
DATAPATH="/pscratch/sd/b/balewski/2025_causalNet_tmp/"
NUM_SAMPLES=200000
N_EPOCHS=300
BATCH_SIZE=2048
LR=1e-3
L1_ALPHA=1e-3
FITNAME=""
DESYNC_TIME=0
TMASK=""
DROPDATAFRAC=0.0

# GPU configuration
NUM_GPUS=4
CUDA_DEVICES="0,1,2,3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataName)
            DATANAME="$2"
            shift 2
            ;;
        --dataPath)
            DATAPATH="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -n|--n_epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --L1_alpha)
            L1_ALPHA="$2"
            shift 2
            ;;
        --fitName)
            FITNAME="$2"
            shift 2
            ;;
        --desync_time)
            DESYNC_TIME="$2"
            shift 2
            ;;
        --Tmask)
            TMASK="--Tmask"
            shift
            ;;
        --dropDataFrac)
            DROPDATAFRAC="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Distributed Lasso Poisson training wrapper script"
            echo ""
            echo "Options:"
            echo "  --dataName NAME       Input data name (default: $DATANAME)"
            echo "  --dataPath PATH       Path to data files (default: $DATAPATH)"
            echo "  --num_samples N       Number of samples to use (default: $NUM_SAMPLES)"
            echo "  -n, --n_epochs N      Number of training epochs (default: $N_EPOCHS)"
            echo "  --batch_size N        Batch size (default: $BATCH_SIZE)"
            echo "  --lr RATE             Learning rate (default: $LR)"
            echo "  --L1_alpha ALPHA      L1 regularization alpha (default: $L1_ALPHA)"
            echo "  --fitName NAME        Fit output name (default: auto-generated)"
            echo "  --desync_time N       Time shift for decorrelation (default: $DESYNC_TIME)"
            echo "  --Tmask               Use time mask to remove time bins"
            echo "  --dropDataFrac FRAC   Fraction of data to drop per rank (default: $DROPDATAFRAC)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "GPU Configuration:"
            echo "  Number of GPUs: $NUM_GPUS"
            echo "  CUDA devices: $CUDA_DEVICES"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build the Python command arguments
PYTHON_ARGS="--dataName $DATANAME --dataPath $DATAPATH --num_samples $NUM_SAMPLES --n_epochs $N_EPOCHS --batch_size $BATCH_SIZE --lr $LR --L1_alpha $L1_ALPHA --desync_time $DESYNC_TIME --dropDataFrac $DROPDATAFRAC"

# Add optional fitName if provided
if [[ -n "$FITNAME" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --fitName $FITNAME"
fi

# Add Tmask flag if set
if [[ -n "$TMASK" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS $TMASK"
fi

# Print configuration
echo "=== Distributed Lasso Poisson Training ==="
echo "Data: $DATANAME"
echo "Epochs: $N_EPOCHS, Samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE, LR: $LR"
echo "L1 alpha: $L1_ALPHA, Drop fraction: $DROPDATAFRAC"
echo "GPUs: $NUM_GPUS ($CUDA_DEVICES)"
echo "==========================================="
echo ""

# Set environment variables and run the training
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Execute the training command
echo "Running: torchrun --standalone --nproc_per_node=$NUM_GPUS ./fit_lassoPoisson.py $PYTHON_ARGS"
echo ""

time torchrun --standalone --nproc_per_node=$NUM_GPUS ./fit_lassoPoisson.py $PYTHON_ARGS
