#!/bin/bash
# Scan FDR recovery versus amount of data.
# Usage:
#   ./big_scanTime_FDR.sh abc 10 30
#
# The second and third arguments are start/stop minutes. prism_EM_train3c.py
# receives this range in seconds through --time_range_sec.

set -euo pipefail
export OMP_NUM_THREADS=1
SECONDS=0

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 SUFFIX START_MIN STOP_MIN"
    echo "Example: $0 abc 10 30"
    exit 2
fi

scanSuffix="$1"
scanStartMin="$2"
scanStopMin="$3"
for val in "$scanStartMin" "$scanStopMin"; do
    if ! [[ "$val" =~ ^[0-9]+$ ]]; then
        echo "ERROR: START_MIN and STOP_MIN must be non-negative integers, got '$scanStartMin' '$scanStopMin'"
        exit 2
    fi
done
if [[ "$scanStopMin" -le "$scanStartMin" ]]; then
    echo "ERROR: STOP_MIN must be greater than START_MIN, got '$scanStartMin' '$scanStopMin'"
    exit 2
fi

module load pytorch
cd "$(dirname "$0")"

# ---------- dataset selection ----------
# Synthetic example
basePath=/pscratch/sd/b/balewski/2026_causalNet_exp_ver3c
#shortN=daleN200_74e6d6_e2b7d7  # N200 12/21 Hz
#shortN=daleN200_f33b3b_7d8ff1  # N200 6/11  Hz
shortN=daleN200_55e5a6_ff089c  # N200 3/21  Hz
numStates=2

# experimental data
#shortN=Canine_260324_r21_w0_1hz; numStates=1 
shortN=Canine_260324_r23_w0_1hz; numStates=2 

timeRange=($((scanStartMin * 60)) $((scanStopMin * 60)))

# ---------- reference EM: state discovery ----------
numEmIters=12
numEMepochs=2
emBatchSize=4096

# ---------- FDR bags: locked A/B fitting ----------
numBags=10
bagFrac=0.8
bagEpochs=150
numScrambles=6
bagBatchSize=4096
perBagQuantile=0.95
stabSelThresh=0.7

runTag="$(python3 -c 'import secrets; print(secrets.token_hex(2))')"

scanTimeTag="${scanStartMin}to${scanStopMin}min"
emFitName="${shortN}_${scanSuffix}_${scanTimeTag}_em${runTag}"
fdrBagsName="${emFitName}_fdr${runTag}"
fdrAgrName="${shortN}_${scanSuffix}_${scanTimeTag}"

echo "basePath=$basePath"
echo "dataName=$shortN"
echo "scanSuffix=$scanSuffix"
echo "scanStartMin=$scanStartMin"
echo "scanStopMin=$scanStopMin"
echo "runTag=$runTag"
echo "timeRange=${timeRange[*]} sec  numStates=$numStates"
echo "emFitName=$emFitName"
echo "fdrBagsName=$fdrBagsName"
echo "fdrAgrName=$fdrAgrName"
echo "perBagQuantile=$perBagQuantile"
echo "stabSelThresh=$stabSelThresh"

echo
echo "=== Reference EM fit: state discovery ==="
time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_EM_train3c.py \
  --basePath "$basePath" \
  --dataName "$shortN" \
  --fitName "$emFitName" \
  --time_range_sec "${timeRange[0]}" "${timeRange[1]}" \
  --num_states "$numStates" \
  --num_em_iters "$numEmIters" \
  --m_epochs "$numEMepochs" \
  --batch_size "$emBatchSize" \
  --delay_em_iter_4_lrDecay 4 "$numEmIters" \
  --delay_em_iter_4_ArhoMax 6 "$numEmIters" \
  --delay_em_iter_4_Aprune 6

echo
echo "=== FDR bags: locked M-step A/B fits ==="
for ((bag=0; bag<numBags; bag++)); do
    printf "\n--- bag %03d/%03d, elapsed %.1f min ---\n" "$bag" "$((numBags - 1))" "$(awk "BEGIN {print $SECONDS/60.0}")"
    time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
      ./prism_FDR_Bags_train3c.py \
      --basePath "$basePath" \
      --emFitName "$emFitName" \
      --outFitName "$fdrBagsName" \
      --bag_idx "$bag" \
      --bag_frac "$bagFrac" \
      --epochs "$bagEpochs" \
      --batch_size "$bagBatchSize" \
      --num_scrambles "$numScrambles"
done

echo
echo "=== Aggregate bags ==="
./prism_EM_FDR_Bags_aggregate3c.py \
  --basePath "$basePath" \
  --dataName "$fdrBagsName" \
  --outAgrName "$fdrAgrName" \
  --num_bags "$numBags" \
  --per_bag_quantile "$perBagQuantile" \
  --stab_sel_thresh "$stabSelThresh"

echo
echo "Done."
echo "Reference EM: $basePath/prismFit/${emFitName}.prismEM.npz"
echo "Bag files: $basePath/prismFDR/${fdrBagsName}.bagNNN.prismFDRbag.npz"
echo "Aggregate: $basePath/prismFit/${fdrAgrName}.prismEM.npz"
