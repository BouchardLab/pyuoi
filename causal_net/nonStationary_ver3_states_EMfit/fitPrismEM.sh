#!/bin/bash
# salloc -q interactive -C gpu  -t 4:00:00  -N 1 -A m2043
#  module load pytorch
#  basePath=/pscratch/sd/b/balewski/2026_causalNet_tmp3/
#  Simplified wrapper script for distributed Lasso Poisson training
#  
# ./fitPrismEM.sh --basePath $basePath  --dataName  daleN100_b6ce1e_2ce4d2 --num_states 2 --num_em_iters 2 --m_epochs 16    --fitName abc16


# Capture variable arguments as text
varArgs="$*"

# Require key variable arguments to be explicitly provided
for required_arg in --dataName --num_states --basePath ; do
    if [[ " $varArgs " != *" $required_arg "* ]]; then
        echo "ERROR: Missing required argument in varArgs: $required_arg" >&2
        echo "Provided varArgs: $varArgs" >&2
        exit 1
    fi
done

# Validate that --basePath points to an existing directory
BASE_PATH=""
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    if [[ "$arg" == "--basePath" ]]; then
        j=$((i + 1))
        if [ "$j" -le "$#" ]; then
            BASE_PATH="${!j}"
        fi
    elif [[ "$arg" == --basePath=* ]]; then
        BASE_PATH="${arg#--basePath=}"
    fi
done

if [ -z "$BASE_PATH" ]; then
    echo "ERROR: --basePath is required and must include a directory value." >&2
    exit 1
fi

if [ ! -d "$BASE_PATH" ]; then
    echo "ERROR: --basePath directory does not exist: $BASE_PATH" >&2
    exit 1
fi

# Default/fixed arguments
fixArgs="  --time_range_sec 0 80 "
# --dataPath /pscratch/sd/b/balewski/2025_causalNet_tmp/

# GPU configuration
NUM_GPUS=4
CUDA_DEVICES="0,1,2,3"

# Ensure enough GPUs are available before launching distributed training
HOST_NAME=$(hostname 2>/dev/null || echo "unknown-host")

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR [$HOST_NAME]: nvidia-smi not found; cannot verify GPU availability." >&2
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "ERROR [$HOST_NAME]: Need at least $NUM_GPUS GPUs, but only $AVAILABLE_GPUS detected." >&2
    exit 1
fi

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
echo "Running: ./prism_EM_train.py  $fixArgs $varArgs"
echo ""

time torchrun --standalone  --nnodes=1 --nproc_per_node=4  ./prism_EM_train.py $fixArgs $varArgs 
