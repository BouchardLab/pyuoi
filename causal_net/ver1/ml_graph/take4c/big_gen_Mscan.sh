#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value

# salloc -q interactive -C cpu -t 4:00:00 -A m2043 -N 1

stop_me 
# Set base path

#basePath=/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp/

K=6  

connProb=0.15
# simulation 
evolT=200_001  # steps

batchSize=256
epochs=200

# Loop to generate 20 matrices
for ((i=0; i<$K; i++)); do
    
    M=$((80+i*40)) # M=40...140
    
    matrixName="daleM${M}jun28"
    simName="${matrixName}_sim" 
    
    echo "Generating matrix ${i} of $K: ${matrixName}"
    
    # Generate the Dale matrix + simu
    ./gen_data.py -T $evolT -M $M --sparse $connProb --simName $simName -X
       
    # fitting
    time ./fit_model.py --input $simName --epochs $epochs --batch $batchSize --lr 0.01 -X

    echo "------------------------ ended i=$i"
done

echo "All $K matrices have been generated"  

#output .npz file: data/dataM20_f30725.npz
#output .png file: data/trajM20_f30725.png
 
