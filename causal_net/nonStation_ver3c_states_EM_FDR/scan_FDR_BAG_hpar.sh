#!/bin/bash
# Scan Stage (b) FDR-bag aggregation hyperparameters for an existing bag set.
#
# This script does not rerun EM or bag fits. It reuses:
#   $basePath/prismFDR/${fdrBagsName}.bagNNN.prismFDRbag.npz
# and writes aggregate fits to:
#   $basePath/prismFit/${outAgrName}_thrX.prismEM.npz
#   $basePath/prismFit/${outAgrName}_qX.prismEM.npz

set -euo pipefail
export OMP_NUM_THREADS=1
SECONDS=0

module load pytorch
cd "$(dirname "$0")"

# ---------- input bag set ----------
basePath=/pscratch/sd/b/balewski/2026_causalNet_exp_ver3c
fdrBagsName=daleN200_74e6d6_e2b7d7_take1_30min_embe98_fdrbe98 # N200 12/21  Hz
fdrBagsName=daleN200_55e5a6_ff089c_take1_60min_em4c42_fdr4c42 # N200 3/6  Hz
outAgrName=aaa24
numBags=11

# ---------- baseline aggregation hyperparameters ----------
base_per_bag_quantile=0.95
base_stab_sel_thresh=0.7

# ---------- scan points ----------
stabSelScan=(0.8 0.7 0.6 0.5 )
perBagQuantileScan=(0.99 0.97 0.95 0.90 0.85)

echo "basePath=$basePath"
echo "fdrBagsName=$fdrBagsName"
echo "outAgrName=$outAgrName"
echo "numBags=$numBags"
echo "base_per_bag_quantile=$base_per_bag_quantile"
echo "base_stab_sel_thresh=$base_stab_sel_thresh"
echo "stabSelScan=${stabSelScan[*]}"
echo "perBagQuantileScan=${perBagQuantileScan[*]}"

echo
echo "=== Scan stability selection threshold ==="
for thr in "${stabSelScan[@]}"; do
    scanOutAgrName="${outAgrName}_thr${thr}"
    printf "\n--- stab_sel_thresh=%s, elapsed %.1f min ---\n" "$thr" "$(awk "BEGIN {print $SECONDS/60.0}")"
    ./prism_EM_FDR_Bags_aggregate3c.py \
      --basePath "$basePath" \
      --dataName "$fdrBagsName" \
      --outAgrName "$scanOutAgrName" \
      --num_bags "$numBags" \
      --per_bag_quantile "$base_per_bag_quantile" \
      --stab_sel_thresh "$thr"
done

echo
echo "=== Scan per-bag null quantile ==="
for q in "${perBagQuantileScan[@]}"; do
    scanOutAgrName="${outAgrName}_q${q}"
    printf "\n--- per_bag_quantile=%s, elapsed %.1f min ---\n" "$q" "$(awk "BEGIN {print $SECONDS/60.0}")"
    ./prism_EM_FDR_Bags_aggregate3c.py \
      --basePath "$basePath" \
      --dataName "$fdrBagsName" \
      --outAgrName "$scanOutAgrName" \
      --num_bags "$numBags" \
      --per_bag_quantile "$q" \
      --stab_sel_thresh "$base_stab_sel_thresh"
done

echo
echo "Done."
echo "Input bags: $basePath/prismFDR/${fdrBagsName}.bagNNN.prismFDRbag.npz"
echo "Aggregates:"
for thr in "${stabSelScan[@]}"; do
    echo "  $basePath/prismFit/${outAgrName}_thr${thr}.prismEM.npz"
done
for q in "${perBagQuantileScan[@]}"; do
    echo "  $basePath/prismFit/${outAgrName}_q${q}.prismEM.npz"
done

thrTags=()
for thr in "${stabSelScan[@]}"; do
    thrTags+=("thr${thr}")
done
qTags=()
for q in "${perBagQuantileScan[@]}"; do
    qTags+=("q${q}")
done

echo
echo "Metric commands:"
echo "./edgeMaterAbs3c.py --basePath \"$basePath\" --fitNameTrunk \"$outAgrName\" --fitTags ${thrTags[*]}"
echo "./edgeMaterAbs3c.py --basePath \"$basePath\" --fitNameTrunk \"$outAgrName\" --fitTags ${qTags[*]}"
