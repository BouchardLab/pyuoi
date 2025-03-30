#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#  salloc -q interactive -C cpu -t 4:00:00 -A m4916    # PI=Katie, NERSC Quantum Repo
#  various configurations of noisy quera +QCrank for QCE25 paper

#stop-me

#basePath="/dataVault2025/paper_QCE2025_"

nshot=${1:-25_000}  
nqa=${2:-3}
nqd=${3:-6}
basePath=${4:-out2}

echo bp=$basePath  shots=$nshot

#nqa=4 ; nshot=50_000  ; nqd=12
#nqa=5 ; nshot=100_000 ; nqd=10

core=qcr${nqa}a+${nqd}d

declare -A cases=(
    ["ideal"]="--idealSimu"
    ["noZo"]="--empty 0"
    ["mZo"]="--multiZone"
    ["1Zo"]="--moveDataQ"
)

declare -A fac_values=(
    ["s07"]="--noiseScale 0.7"
    ["s13"]="--noiseScale 1.3"
)


for tag1 in "${!cases[@]}"; do
    opt1="${cases[$tag1]}"
    echo "name=$tag1  task=$opt1"
    expName=${core}_${tag1}
    
    comm="  --outPath $basePath --numQubits ${nqa} ${nqd}  --numSample 50 --numShot $nshot $opt1 -p ab "
    ./noisy_qcrankV3.py $comm -E --expName $expName  >&${basePath}/log.$expName


    if [ "$tag1" == "ideal" ]; then
        continue
    fi

    for tag2  in "${!fac_values[@]}"; do
	opt2="${fac_values[$tag2]}"
	echo "$tag2  value=$opt2"
	expName2=${expName}_${tag2}
	./noisy_qcrankV3.py $comm $opt2 -E --expName $expName2  >&${basePath}/log.$expName2	
    done
    
done
