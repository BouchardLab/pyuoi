#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value

#stop-me

basePath="/pscratch/sd/b/balewski/2026_causalNet_tmp4topo/"

N=200
L=10  # 1,2,5,10,20

for dker in $(seq -f "%.2f" 0.10 0.05 2.80); do

    expName=aN${N}_L${L}_dker${dker}
    #echo expName=$expName  

    #./gen_daleMatrices4.py --spike_model A  --basePath $basePath --placement_H_L_delta 1 $L $dker  --num_neurons $N --num_steps 1001  --dataName $expName
    ./topoAna_daleMatrix4.py  --basePath $basePath   --dataName $expName |grep \#V
    #break
done

echo all DONE
 
