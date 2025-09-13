#!/bin/bash
#
# bigFitLasso.sh - Bootstrap wrapper for distributed Lasso Poisson training
#
# Usage: ./bigFitLasso.sh [OPTIONS]
#
# This script runs fitLasso.sh multiple times for bootstrap training

# Default values
DATANAME="daleM80_285c84"
N_EPOCHS=300
NUM_SAMPLES="350_000"
DROPDATAFRAC=0.5
NUM_BOOTSTRAPS=5

# Other arguments to pass through to fitLasso.sh
OTHER_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataName)
            DATANAME="$2"
            shift 2
            ;;
        -n|--n_epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --dropDataFrac)
            DROPDATAFRAC="$2"
            shift 2
            ;;
        --num_bootstraps)
            NUM_BOOTSTRAPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Bootstrap wrapper for distributed Lasso Poisson training"
            echo ""
            echo "Bootstrap-specific options:"
            echo "  --dataName NAME       Input data name (default: $DATANAME)"
            echo "  -n, --n_epochs N      Number of training epochs (default: $N_EPOCHS)"
            echo "  --num_samples N       Number of samples to use (default: $NUM_SAMPLES)"
            echo "  --dropDataFrac FRAC   Fraction of data to drop per rank (default: $DROPDATAFRAC)"
            echo "  --num_bootstraps N    Number of bootstrap iterations (default: $NUM_BOOTSTRAPS)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "All other options are passed through to fitLasso.sh"
            echo ""
            echo "Each bootstrap run will use fitName: \${dataName}_boot\${k}"
            echo ""
            exit 0
            ;;
        *)
            # Collect all other arguments to pass to fitLasso.sh
            OTHER_ARGS="$OTHER_ARGS $1"
            if [[ $2 && $2 != --* ]]; then
                OTHER_ARGS="$OTHER_ARGS $2"
                shift 2
            else
                shift
            fi
            ;;
    esac
done

# Validation
if [[ $NUM_BOOTSTRAPS -lt 1 ]]; then
    echo "Error: num_bootstraps must be at least 1"
    exit 1
fi

if [[ ! -f "./fitLasso.sh" ]]; then
    echo "Error: fitLasso.sh not found in current directory"
    exit 1
fi

# Print configuration
echo "=== Bootstrap Lasso Poisson Training ==="
echo "Data: $DATANAME"
echo "Bootstraps: $NUM_BOOTSTRAPS"
echo "Epochs: $N_EPOCHS, Samples: $NUM_SAMPLES"
echo "Drop fraction: $DROPDATAFRAC"
echo "Other args: $OTHER_ARGS"
echo "========================================="
echo ""

# Record start time
TOTAL_START_TIME=$(date +%s)

# Bootstrap loop
for ((k=1; k<=NUM_BOOTSTRAPS; k++)); do
    # Construct fitName
    FITNAME="${DATANAME}-boot${k}"
    
    echo "=== Bootstrap $k/$NUM_BOOTSTRAPS ==="
    echo "fitName: $FITNAME"
    echo "Starting at: $(date)"
    echo ""
    
    # Record individual run start time
    RUN_START_TIME=$(date +%s)
    
    # Run fitLasso.sh with constructed fitName and all parameters
    ./fitLasso.sh \
        --dataName "$DATANAME" \
        --n_epochs "$N_EPOCHS" \
        --num_samples "$NUM_SAMPLES" \
        --dropDataFrac "$DROPDATAFRAC" \
        --fitName "$FITNAME" \
        $OTHER_ARGS
    
    # Check exit status
    EXIT_CODE=$?
    
    # Record individual run end time
    RUN_END_TIME=$(date +%s)
    RUN_DURATION=$((RUN_END_TIME - RUN_START_TIME))
    
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "=== Bootstrap $k/$NUM_BOOTSTRAPS COMPLETED ==="
        echo "Duration: ${RUN_DURATION}s ($(date -d@$RUN_DURATION -u +%H:%M:%S))"
        echo "Finished at: $(date)"
        echo ""
    else
        echo ""
        echo "=== Bootstrap $k/$NUM_BOOTSTRAPS FAILED ==="
        echo "Exit code: $EXIT_CODE"
        echo "Duration: ${RUN_DURATION}s"
        echo ""
        
        # Ask user if they want to continue
        read -p "Continue with remaining bootstraps? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping bootstrap execution."
            exit $EXIT_CODE
        fi
        echo ""
    fi
    
    # Add separator between runs (except after last run)
    if [[ $k -lt $NUM_BOOTSTRAPS ]]; then
        echo "################################################################"
        echo ""
    fi
done

# Record total end time and calculate duration
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo "################################################################"
echo "=== ALL BOOTSTRAPS COMPLETED ==="
echo "Total bootstraps: $NUM_BOOTSTRAPS"
echo "Total time: ${TOTAL_DURATION}s (${HOURS}h ${MINUTES}m ${SECONDS}s)"
echo "Average per bootstrap: $((TOTAL_DURATION / NUM_BOOTSTRAPS))s"
echo "Finished at: $(date)"
echo "################################################################"

echo ""
echo "Bootstrap results saved with fitNames:"
for ((k=1; k<=NUM_BOOTSTRAPS; k++)); do
    echo "  ${DATANAME}-boot${k}"
done

echo ""
echo "To evaluate the first bootstrap result, run:"
echo "./eval_fitLasso.py --dataPath /pscratch/sd/b/balewski/2025_causalNet_tmp/ --dataName ${DATANAME}-boot1 -p a"
