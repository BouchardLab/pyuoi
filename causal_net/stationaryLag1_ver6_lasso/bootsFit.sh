#!/bin/bash
#
# boostFit.sh - Simplified bootstrap wrapper for distributed Lasso Poisson training
#
# This script runs fitLasso.sh multiple times for bootstrap training

# Parse arguments with better handling
DATANAME=""  # mandatory parameter
NUM_BOOTSTRAPS=6  # default value
SHUFFLE_TIME=false
BOOTS_TAG='s1'
varArgs=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_name|--dataName)
            DATANAME="$2"
            varArgs+=("--dataName" "$2")
            shift 2
            ;;
        --num_bootstraps)
            NUM_BOOTSTRAPS="$2"
            shift 2
            ;;
        --bootsTag)
            BOOTS_TAG="$2"
            shift 2
            ;;
        --desyncTime)
            DESYNC_TIME=true
            varArgs+=("--desyncTime")
            shift
            ;;
        *)
            # Pass through all other arguments
            varArgs+=("$1")
            shift
            ;;
    esac
done

# Validate mandatory parameter
if [[ -z "$DATANAME" ]]; then
    echo "Error: --data_name (or --dataName) is mandatory"
    echo "Usage: $0 --data_name <name> [--num_bootstraps <number>] [--shuffleTime] [other_args...]"
    exit 1
fi
echo ${BOOTS_TAG}

# Set naming pattern based on shuffleTime flag
FIT_SUFFIX="boots"
if [[ "$DESYNC_TIME" == true ]]; then
    FIT_SUFFIX="desync"
fi

# Convert varArgs array back to string for compatibility
varArgsStr="${varArgs[*]}"

# Default/fixed arguments (removed --dropDataFrac since it can be passed via command line)
fixArgs=""

# Validation
if [[ ! -f "./fitLasso4GPU.sh" ]]; then
    echo "Error: fitLasso4GPU.sh not found in current directory"
    exit 1
fi

# Print configuration
echo "=== Bootstrap Lasso Poisson Training ==="
echo "Extracted dataName: $DATANAME"
echo "FitName pattern: ${DATANAME}${BOOTS_TAG}-${FIT_SUFFIX}X (starting from 0)"
echo "Variable args: $varArgsStr"
echo "Default args: $fixArgs"
echo "Bootstraps: $NUM_BOOTSTRAPS"
echo "========================================="
echo ""

# Record start time
TOTAL_START_TIME=$(date +%s)

# Bootstrap loop
for ((k=1; k<=NUM_BOOTSTRAPS; k++)); do
    # Construct fitName using extracted dataName and suffix (count from 0)
    FITNAME="${DATANAME}${BOOTS_TAG}-${FIT_SUFFIX}$((k-1))"
    
    echo "=== Bootstrap $k/$NUM_BOOTSTRAPS ==="
    echo "fitName: $FITNAME"
    echo "Starting at: $(date)"
    echo ""
    
    # Record individual run start time
    RUN_START_TIME=$(date +%s)
    
    # Run fitLasso.sh with simplified approach
    ./fitLasso4GPU.sh $fixArgs --fitName "$FITNAME" $varArgsStr
   
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
echo " ./selectEdges_FDR.py  --dataPath \$fitPath --dataName  ${DATANAME}${BOOTS_TAG}   --num_bootstraps  $NUM_BOOTSTRAPS --alphaFDR 1e-3 -p ab  "
echo "################################################################"

echo ""
echo "Bootstrap results saved with fitNames:"
for ((k=1; k<=NUM_BOOTSTRAPS; k++)); do
    echo "  ${DATANAME}-${FIT_SUFFIX}$((k-1))"
done


# ./fit_lassoPoisson.py $fixArgs $varArgs   # 1 GPU job, for testing


#  ./fitLasso4GPU.sh   --outPath $fitPath --inpPath $inpPath   --dataName daleM140r1Hz  --num_epochs 50   --num_samples 100_001

#  ./bootsFit.sh   --outPath $fitPath --inpPath $inpPath   --dataName daleM140r1Hz  --num_epochs 50   --num_samples 100_001  --dropDataFrac 0.33  --num_bootstraps 1   --bootsTag b1
