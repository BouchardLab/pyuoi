#!/bin/bash

# Set base path
basePath=/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp/


# simulation 
evolT=60  # seconds
matrixName=daleM140may27

#sigL=" 1 4 16 32 64 "

sigL=" 256 1024 "

for sig in $sigL ; do
    echo $sig
    simuName=${matrixName}_sim_sig$sig
    echo "simu" $simuName
    #./simu_netActivity.py  --basePath $basePath  --matrixName ${matrixName}   --evol_time $evolT --sigma_noise $sig --outName  $simuName 


    #format data for fitting
    #../fit_UoI/prep_simInput.py --basePath $basePath  --simName $simuName    

    srun -n512 --distribution=block:block shifter python  fit_uoiVar_admm.py  --basePath $basePath   --inpName    $simuName  --num_admm 8  --time_range 0. 10.  

    echo "------------------------"
    
done

echo "All $K matrices have been generated"  
