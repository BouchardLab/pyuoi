#!/bin/bash
# Example end-to-end PRISM-EM FDR bagging run.
# salloc -q interactive -C gpu -t 4:00:00 -N 1 -A m2043

set -euo pipefail
export OMP_NUM_THREADS=1
SECONDS=0

module load pytorch
cd "$(dirname "$0")"

# ---------- dataset selection ----------
# Synthetic example
#basePath=/pscratch/sd/b/balewski/2026_causalNet_exp_ver3c
#shortN=daleN200_55e5a6_ff089c  # N200 3/21  Hz
#shortN=daleN200_f33b3b_7d8ff1  # N200 6/11  Hz
#shortN=daleN200_74e6d6_e2b7d7  # N200 12/21 Hz
#shortN=daleN100_d4f303_4abc4c  # N100 13/27 Hz
#numStates=2

# Experimental example
basePath=/pscratch/sd/b/balewski/2026_causalNet_exp_ver3c
#shortN=Canine_260324_r23_w0_1hz; numStates=2
shortN=Canine_260324_r21_w0_1hz; numStates=1
#timeRange=(0 3600)
#timeRange=(0 1800)
timeRange=(1800 3600)
#timeRange=(0 300)

# ---------- reference EM: state discovery ----------
numEmIters=12
numEMepochs=2
emBatchSize=4096

# ---------- FDR bags: locked A/B fitting ----------
numBags=11
bagFrac=0.8
bagEpochs=180
numScrambles=6
bagBatchSize=4096
runAggregate=1

runTag="$(python3 -c 'import secrets; print(secrets.token_hex(2))')"

echo "basePath=$basePath"
echo "dataName=$shortN"
echo "runTag=$runTag"
echo "timeRange=${timeRange[*]}  numStates=$numStates"
emFitName="${shortN}_em${runTag}"
fdrBagsName="${emFitName}_fdr${runTag}"
fdrAgrName="${fdrBagsName}_agr${runTag}"

echo "emFitName=$emFitName"
echo "fdrBagsName=$fdrBagsName"
echo "fdrAgrName=$fdrAgrName"

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

if [[ "$runAggregate" == "1" ]]; then
    echo
    echo "=== Aggregate bags ==="
      ./prism_EM_FDR_Bags_aggregate3c.py \
      --basePath "$basePath" \
      --dataName "$fdrBagsName" \
      --outAgrName "$fdrAgrName" \
      --num_bags "$numBags" \
      --per_bag_quantile 0.97 \
      --stab_sel_thresh 0.7
fi

echo
echo "Done."
echo "Bag files: $basePath/prismFDR/${fdrBagsName}.bagNNN.prismFDRbag.npz"
if [[ "$runAggregate" == "1" ]]; then
    echo "Aggregate: $basePath/prismFit/${fdrAgrName}.prismEM.npz"
fi
