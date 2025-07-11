#!/bin/bash

# Set base path
basePath=/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp/

K=10 
N=20
M=$((2*$N))  # Total neurons (N excitatory + N inhibitory)
connProb=0.15
# simulation 
evolT=20  # seconds
sigma=2.0

# Loop to generate 20 matrices
for ((i=0; i<$K; i++)); do
    # Create unique name using date and iteration number
    matrixName="daleM${M}may26_${i}"
    simuName="${matrixName}_simu" 
    
    echo "Generating matrix ${i} of 20: ${matrixName}"
    
    # Generate the Dale matrix
    ./gen_daleMatrix.py  --basePath $basePath  \
        --num_excit_neur $N \
        --matrixName $matrixName \
        --prob_synaptic_conn $connProb
    
    # simulate net response
     ``./simu_netActivity.py  --basePath $basePath  --matrixName $matrixName --outName $simuName \
     --evol_time $evolT --sigma_noise $sigma 
    echo "Completed matrix ${i}"

    #format data for fitting
    ../fit_UoI/prep_simInput.py --basePath $basePath  --simName $simuName  --saveNPY $i

    echo "------------------------"
done

echo "All $K matrices have been generated"  