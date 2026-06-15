# PRISM-EM 3c Training

This directory contains the lag-1 PRISM-EM trainer and the Stage (a)
EM-FDR-bagging trainer.

The two entry points share the same core implementation in
`PrismEM_Workhorse3c.py`:

- `prism_EM_train3c.py`: ordinary time-ordered EM fit.
- `prism_EM_FDR_Bags_train3c.py`: one EM-FDR-bagging Stage (a) job, i.e.
  one resampled bag plus its circular-shift null refits.
- `prism_EM_FDR_Bags_agregate3c.py`: EM-FDR-bagging Stage (b), i.e.
  aggregate all bag files into one eval-compatible fit.

The code expects input spike files under:

```bash
$basePath/spikesData/<dataName>.spikes.npz
```

On Perlmutter, load PyTorch before running:

```bash
module load pytorch
```

For reproducible multi-GPU runs, it is also useful to cap CPU thread pools:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## Plain EM Fit

Run the ordinary time-ordered EM fit with `torchrun`. This writes one fit file
to:

```bash
$basePath/prismFit/<fitName>.prismEM.npz
```

Example using 4 GPUs:

```bash
basePath=/path/to/run
shortN=myDataset

time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_EM_train3c.py \
  --basePath $basePath \
  --dataName $shortN \
  --num_states 2 \
  --num_em_iters 50 \
  --batch_size 4096 \
  --delay_em_iter_4_lrDecay 5 \
  --delay_em_iter_4_ArhoMax 20 \
  --delay_em_iter_4_Aprune 30 \
  --time_range_sec 300 900
```

Useful optional controls:

- `--fitName`: output stem. If omitted, the trainer appends a short random
  suffix.
- `--init_states {data,rand}`: initialize state probabilities from the data
  or randomly.
- `--init_A {data,rand}`: initialize lag-1 connectivity from spike data or
  randomly.
- `--init_B {data,rand}`: initialize state biases from spike rates or
  randomly.
- `--init_A_Tmax`: maximum number of time bins used for data-driven `A`
  initialization.

Training defaults used by both entry points include `--m_epochs 2`,
`--pgd_iter 5`, and `--prescale_m_step_4_ArhoMax 120`.

## EM-FDR-Bags Stage (a)

Run `prism_EM_FDR_Bags_train3c.py` once per bag. Each invocation:

1. Draws `--num_blocks` contiguous blocks from `--time_range_sec`.
2. Concatenates them into one bag.
3. Runs a real, time-ordered PRISM-EM fit on that bag.
4. Freezes the decoded state sequence `S_hat`.
5. Runs `--num_scrambles` circular-shift null refits using locked
   `S_hat[t-1]` for pairs `(Y[t-1], Y[t])`.
6. Writes one self-contained bag file.

The output file is:

```bash
$basePath/prismFDR/<dataName>_<bagsTag>.bag<bag_idx:03d>.prismFDRbag.npz
```

Pass `--bagsTag TAG` to choose the tag. If `--bagsTag` is omitted or set to
`None`, Stage (a) derives a deterministic four-character alphanumeric tag
from the shared bagging/training configuration, excluding `bag_idx`, so array
tasks for different bags use the same tag.

For `--bag_idx 0`, blocks are assembled chronologically on the input-data
timeline: block 0 starts at the beginning of `--time_range_sec`, block 1
starts one block length later, and so on. Other bag indices use the random
block-bootstrap sampler.

Example using 4 GPUs for bag 3:

```bash
basePath=/path/to/run
shortN=myDataset

time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_EM_FDR_Bags_train3c.py \
  --basePath $basePath \
  --dataName $shortN \
  --bag_idx 3 \
  --bagsTag testA \
  --time_range_sec 300 900 \
  --num_blocks 10 \
  --num_scrambles 4 \
  --num_states 2 \
  --num_em_iters 50 \
  --batch_size 4096 \
  --delay_em_iter_4_lrDecay 5 \
  --delay_em_iter_4_ArhoMax 20 \
  --delay_em_iter_4_Aprune 30
```

The bag driver inherits the ordinary EM controls and adds these FDR-bagging
controls:

- `--bag_idx`: integer bag index used in the output filename and RNG stream.
- `--bagsTag TAG`: alphanumeric tag appended after `dataName` in Stage (a)
  output names. Use `None` or omit it for the deterministic four-character tag.
- `--fdr_out_dir`: optional output directory; default is `$basePath/prismFDR`.
- `--time_range_sec START END`: time range for block starts.
- `--num_blocks`: number of contiguous blocks in the bag.
- `--block_len_sec`: block length in seconds; default `60`.
- `--min_block_start_sep_sec`: preferred minimum separation between block
  starts; default `1`.
- `--max_block_draw_trials`: number of attempts to satisfy the separation
  constraint before accepting the next draw and recording the violation;
  default `30`.
- `--num_scrambles`: number of circular-shift null refits.
- `--min_roll_shift_sec`: exclusion zone around zero shift for each neuron;
  default `2`.
- `--null_A_init {scrambled_data,rand}`: fresh `A` initialization mode for
  each null refit.
- `--null_B_init {scrambled_data,rand}`: fresh `B` initialization mode for
  each null refit.

There is no separate `--null_m_epochs`: null refits inherit the real fit's
M-step epoch count and optimizer configuration. During null refits, pruning,
spectral projection, and learning-rate decay are active from the first locked
M-step epoch.

## Running Multiple Bags

Stage (a) bags are independent jobs. For a small manual launch, vary
`--bag_idx`:

```bash
for bag in 0 1 2 3 4; do
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    ./prism_EM_FDR_Bags_train3c.py \
    --basePath $basePath \
    --dataName $shortN \
    --bag_idx $bag \
    --time_range_sec 300 900 \
    --num_blocks 10 \
    --num_scrambles 4 \
    --num_states 2 \
    --num_em_iters 50
done
```

For production, submit one SLURM job per `bag_idx`, each using one node and
4 GPUs. The bag trainer itself does not perform final pooled FDR selection.

## EM-FDR-Bags Stage (b)

After Stage (a) finishes for `bag000` through `bag<num_bags-1>`, aggregate
the bag files with:

```bash
basePath=/path/to/run
shortN=myDataset
bagsTag=testA
num_bags=5

./prism_EM_FDR_Bags_agregate3c.py \
  --basePath $basePath \
  --dataName ${shortN}_${bagsTag} \
  --num_bags $num_bags \
  --per_bag_quantile 0.99 \
  --sel_prob 0.7
```

The aggregator reads:

```bash
$basePath/prismFDR/<dataName>.bag000.prismFDRbag.npz
$basePath/prismFDR/<dataName>.bag001.prismFDRbag.npz
...
```

and writes:

```bash
$basePath/prismFit/<dataName>_bags<num_bags>.prismEM.npz
```

For example, with `shortN=daleN100_2ba29b_c47b43` and `num_bags=5`, the
output stem is:

```bash
daleN100_2ba29b_c47b43_bags5
```

The output keeps the normal PRISM-EM fields expected by `prism_EM_eval3c.py`.
The connectivity fields are aggregated Stage (b) results, while the time
series fields (`S_hat`, `c_hat`, `S_hat_CL`, and EM histories) are copied
from reference bag 0 for plot compatibility.

You can then run:

```bash
./prism_EM_eval3c.py \
  --basePath $basePath \
  --dataName ${shortN}_bags${num_bags} \
  -p a e f g
```

To inspect one Stage (a) time-ordered bag fit directly, pass the bag stem as
`--dataName`; names containing `bagNNN` are loaded from `prismFDR`:

```bash
./prism_EM_eval3c.py \
  --basePath $basePath \
  --dataName ${shortN}.bag000 \
  -p a e f g
```

Stage (b) also saves FDR diagnostics such as `selection_frequency`,
`selected_mask`, `A_mean_selected`, `A_sd_selected`, `A_mean_all`, `A_sd_all`,
`row_null_mean`, `row_null_sd`, `row_null_tau_mean`, `z_null`, and a compact
selected-edge table (`edge_i`, `edge_j`, `edge_sel_freq`, `edge_A_mean`,
`edge_A_sd_boot`, `edge_z_null`).

## Saved Bag Contents

Each `.prismFDRbag.npz` contains the real fit fields in the same style as
`prism_EM_train3c.py`, plus FDR-bagging fields including:

- `A_null`: null connectivity stack with shape `(num_scrambles, N, N)`.
- `B_null`: null bias stack with shape `(num_scrambles, num_states, N)`.
- `roll_shifts_bin`: realized circular shifts per scramble and neuron.
- `block_start_bins` and `block_start_sec`: ordered block starts used to
  assemble the bag.
- `boundary_pair_indices`: lag-1 pair indices crossing block boundaries.
- `block_retry_violations`: accepted starts that violated the separation
  constraint after exhausting `--max_block_draw_trials`.
- `null_*_epoch`: locked M-step histories for the null refits.

Metadata from the bagging workflow are grouped by stage. `bagsFDR_stageA`
contains the bag assembly settings, null-refit settings, per-scramble null
initialization diagnostics, and `real_fit.{train,init_A,init_B,init_state,
states_recovery_eval}` for the real fit on that bag. The aggregate file also
adds `bagsFDR_stageB` for the cross-bag selection and output settings.

The assembled bag spike train is not saved. It can be reconstructed from the
original spike file and the ordered block starts.
