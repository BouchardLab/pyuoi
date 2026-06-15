#!/bin/bash
# salloc -q interactive -C gpu  -t 4:00:00  -N 1 -A m2043
#  module load pytorch

basePath=/pscratch/sd/b/balewski/2026_causalNet_exp_ver3c
shortN=daleN100_2ba29b_c47b43

for bag in 3 4 5 ; do
#for bag in 0 1 2 ; do
    time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_EM_FDR_Bags_train3c.py \
  --basePath $basePath \
  --dataName $shortN \
  --bag_idx $bag \
  --source_window_sec 300 1200 \
  --num_blocks 10 \
  --num_scrambles 4 \
  --num_states 2 \
  --num_em_iters 5 \
  --batch_size 4096 \
  --delay_em_iter_4_lrDecay 1 \
  --delay_em_iter_4_ArhoMax 3 \
  --delay_em_iter_4_Aprune 4
done
