#!/bin/bash
#
# fitLasso.sh - Simplified wrapper script for distributed Lasso Poisson training
#
# This script runs fit_lassoPoisson.py with optimized multi-GPU settings

# Capture variable arguments as text
varArgs="$*"

# Default/fixed arguments
fixArgs="  --batch_size 2048 --lr 1e-3 --L1_alpha 1e-3 "
# --dataPath /pscratch/sd/b/balewski/2025_causalNet_tmp/

# GPU configuration
NUM_GPUS=4
CUDA_DEVICES="0,1,2,3"

# Print configuration
echo "=== Distributed Lasso Poisson Training ==="
echo "Variable args: $varArgs"
echo "Fixed args: $fixArgs"
echo "GPUs: $NUM_GPUS ($CUDA_DEVICES)"
echo "==========================================="
echo ""

# Set environment variables and run the training
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Execute the training command
echo "Running: ./fit_lassoPoisson.py  $fixArgs $varArgs"
echo ""

time torchrun --standalone --nproc_per_node=4  ./fit_lassoPoisson.py $fixArgs $varArgs 
# ./fit_lassoPoisson.py $fixArgs $varArgs   # 1 GPU job, for testing
