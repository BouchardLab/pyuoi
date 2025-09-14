#!/bin/bash
#
# boostFit.sh - Simplified bootstrap wrapper for distributed Lasso Poisson training
#
# This script runs fitLasso.sh multiple times for bootstrap training

# Capture variable arguments as text
varArgs="$*"

# Extract dataName from varArgs, fallback to default if not found
DATANAME="daleM80x_285c84"  # default value
if [[ $varArgs =~ --dataName[[:space:]]+([^[:space:]]+) ]]; then
    DATANAME="${BASH_REMATCH[1]}"
fi

# Check if --shuffleTime flag is present and set naming pattern accordingly
FIT_SUFFIX="boot"
if [[ $varArgs =~ --shuffleTime ]]; then
    FIT_SUFFIX="shuf"
fi

# Default/fixed arguments
fixArgs="  --dropDataFrac 0.5"
NUM_BOOTSTRAPS=5

# Validation
if [[ ! -f "./fitLasso4GPU.sh" ]]; then
    echo "Error: fitLasso4GPU.sh not found in current directory"
    exit 1
fi

# Print configuration
echo "=== Bootstrap Lasso Poisson Training ==="
echo "Extracted dataName: $DATANAME"
echo "FitName pattern: ${DATANAME}-${FIT_SUFFIX}X (starting from 0)"
echo "Variable args: $varArgs"
echo "Fixed args: $fixArgs"
echo "Bootstraps: $NUM_BOOTSTRAPS"
echo "========================================="
echo ""

# Record start time
TOTAL_START_TIME=$(date +%s)

# Bootstrap loop
for ((k=1; k<=NUM_BOOTSTRAPS; k++)); do
    # Construct fitName using extracted dataName and suffix (count from 0)
    FITNAME="${DATANAME}-${FIT_SUFFIX}$((k-1))"
    
    echo "=== Bootstrap $k/$NUM_BOOTSTRAPS ==="
    echo "fitName: $FITNAME"
    echo "Starting at: $(date)"
    echo ""
    
    # Record individual run start time
    RUN_START_TIME=$(date +%s)
    
    # Run fitLasso.sh with simplified approach
    ./fitLasso4GPU.sh $fixArgs --fitName "$FITNAME" $varArgs
    
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
    echo "  ${DATANAME}-${FIT_SUFFIX}$((k-1))"
done

echo ""
echo "To evaluate the first bootstrap result, run:"
echo "./eval_fitLasso.py --dataPath /pscratch/sd/b/balewski/2025_causalNet_tmp/ --dataName ${DATANAME}-${FIT_SUFFIX}0 -p a"
