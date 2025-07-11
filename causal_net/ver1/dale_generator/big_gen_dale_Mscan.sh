#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value

# salloc -q interactive -C cpu -t 4:00:00 -A m2043 -N 1

stop_me 
# Set base path
#basePath=/global/cfs/cdirs/m2043/causal_inference/dale-gen-sim-may27
basePath=/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp/

K=1  

connProb=0.15
# simulation 
sigma=5.0
evolT=90.5  # seconds

# Loop to generate 20 matrices
for ((i=0; i<$K; i++)); do
    
    N=$((15+i*20)) # M=40...140
    #N=$((120+i*20))  # M>=160
    M=$((2*N))  # Total neurons (N excitatory + N inhibitory)
    #1evolT=${M}.5  # seconds
    # Create unique name using date and iteration number
    matrixName="daleM${M}may27"
    simuName="${matrixName}_simu" 
    
    echo "Generating matrix ${i} of $K: ${matrixName}"
    
    # Generate the Dale matrix
    ./gen_daleMatrix.py  --basePath $basePath  \
        --num_excit_neur $N \
        --matrixName $matrixName \
        --prob_synaptic_conn $connProb
    
    ./plot_daleMatrix.py  --basePath $basePath  --matrixName   $matrixName -p abc    
    echo "Completed matrix ${i}"
    # simulate net response
     ``./simu_netActivity.py  --basePath $basePath  --matrixName $matrixName --outName $simuName \
     --evol_time $evolT --sigma_noise $sigma 
    
    ./plot_simNetActivity.py  --basePath $basePath   --simName   $simuName -p a b e      --time_range 0.5 2.5  

    #format data for fitting
    ../fit_UoI/prep_simInput.py --basePath $basePath  --simName $simuName --time_start 0.5 --saveNPY $simuName
    #spare: 

    echo "------------------------"
done

echo "All $K matrices have been generated"  
