#!/bin/bash
# Example end-to-end PRISM-EM FDR bagging + de-bias run.
# salloc -q interactive -C gpu -t 4:00:00 -N 1 -A m2043

set -euo pipefail
export OMP_NUM_THREADS=1
SECONDS=0

cd "$(dirname "$0")"

if (( $# > 1 )); then
    echo "Usage: $0 [0to1h|1to2h|2to3h|3to4h|0to2h|2to4h|0to4h]" >&2
    exit 2
fi

timeWindow="${1:-0to1h}"
case "$timeWindow" in
    0to1h) timeRange=(0 3600) ;;
    1to2h) timeRange=(3600 7200) ;;
    2to3h) timeRange=(7200 10800) ;;
    3to4h) timeRange=(10800 14400) ;;
    0to2h) timeRange=(0 7200) ;;
    2to4h) timeRange=(7200 14400) ;;
    0to4h) timeRange=(0 14400) ;;
    *)
        echo "ERROR: invalid time window '$timeWindow'" >&2
        echo "Allowed values: 0to1h 1to2h 2to3h 3to4h 0to2h 2to4h 0to4h" >&2
        exit 2
        ;;
esac

module load pytorch
randomTag="$(python3 -c 'import secrets; print(secrets.token_hex(2))')"
runTag="${timeWindow}_em${randomTag}"

# ---------- dataset selection ----------
# Synthetic example
#basePath=/pscratch/sd/b/balewski/2026_causalNet_Aug15
#shortN=daleN200_2290a6_b16fce ; numStates=2  # N200 6/11  Hz

# Experimental example
basePath=/pscratch/sd/b/balewski/2026_causalNet_Aug23
shortN=MouseKCL_260729_w5_r18_1hz; numStates=2; decode_dwell_sec=0.10
#shortN=Canine_260324_r21_w0_1hz; numStates=1

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

# ---------- de-bias fit: refit the aggregate FDR support ----------
runDebias=1
debiasStateMode=locked
debiasEpochs=$bagEpochs
debiasBatchSize=$bagBatchSize


echo "basePath=$basePath"
echo "dataName=$shortN"
echo "runTag=$runTag"
echo "timeRange=${timeRange[*]}  numStates=$numStates  decode_dwell_sec=$decode_dwell_sec"
emFitName="${shortN}_${runTag}"
fdrBagsName="${emFitName}_fdr"
fdrAgrName="${fdrBagsName}_agr"
debiasFitName="${fdrAgrName}_deb"

echo "emFitName=$emFitName"
echo "fdrBagsName=$fdrBagsName"
echo "fdrAgrName=$fdrAgrName"
echo "debiasFitName=$debiasFitName"

if [[ "$runDebias" == "1" && "$runAggregate" != "1" ]]; then
    echo "ERROR: runDebias=1 requires runAggregate=1" >&2
    exit 2
fi

echo
echo "=== Reference EM fit: state discovery ==="
time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_EM_train3c.py \
  --basePath "$basePath" \
  --dataName "$shortN" \
  --fitName "$emFitName" \
  --time_range_sec "${timeRange[0]}" "${timeRange[1]}" \
  --num_states "$numStates" \
  --decode_dwell_sec "$decode_dwell_sec" \
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

if [[ "$runDebias" == "1" ]]; then
    echo
    echo "=== De-biased refit of aggregate support ==="
    time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
      ./prism_deBiasFit3c.py \
      --basePath "$basePath" \
      --fdrFitName "$fdrAgrName" \
      --outFitName "$debiasFitName" \
      --state_mode "$debiasStateMode" \
      --decode_dwell_sec "$decode_dwell_sec" \
      --m_epochs "$debiasEpochs" \
      --batch_size "$debiasBatchSize"
fi

echo
echo "Done."
echo "Bag files: $basePath/prismFDR/${fdrBagsName}.bagNNN.prismFDRbag.npz"
if [[ "$runAggregate" == "1" ]]; then
    echo "Aggregate: $basePath/prismFit/${fdrAgrName}.prismEM.npz"
fi
if [[ "$runDebias" == "1" ]]; then
    echo "De-biased: $basePath/prismFit/${debiasFitName}.prismEM.npz"
fi
