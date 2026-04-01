#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value

#stop-me

basePath="/pscratch/sd/b/balewski/2026_causalNet_tmp4/"

N=100
L=1

#for dker in 0.1 0.5 1.0 1.5 2.0 ; do
for dker in 0.1 1.0 2.0 ; do
    expName=aN${N}_L${L}_dker${dker}
    #echo expName=$expName  

    #./gen_daleMatrices4.py --spike_model B --time_kernel_q_tau    1.5  0.4 --basePath $basePath --placement_H_L_delta 1 $L $dker  --num_neurons 200 --num_steps 8001  --dataName $expName
    ./topoAna_daleMatrix4.py  --basePath $basePath   --dataName $expName |grep \#V
done

echo all DONE
 

