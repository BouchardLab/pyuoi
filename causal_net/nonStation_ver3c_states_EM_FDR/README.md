# PRISM-EM 3c Training

This directory contains the lag-1 PRISM-EM trainer and the FDR bagging
fitters.

The two entry points share the same core implementation in
`PrismEM_Workhorse3c.py`:

- `prism_EM_train3c.py`: ordinary time-ordered EM fit used for state
  discovery.
- `prism_FDR_Bags_train3c.py`: one FDR-bagging Stage (a) job, i.e. one
  reference-locked A/B refit plus its per-neuron time-shuffle null refits.
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

- `--fitName`: output stem. If omitted, the trainer writes
  `<dataName>_<hash4>` with a random four-character hex suffix.
- `--init_states {data,rand}`: initialize state probabilities from the data
  or randomly.
- `--init_A {data,rand}`: initialize lag-1 connectivity from spike data or
  randomly.
- `--init_B {data,rand}`: initialize state biases from spike rates or
  randomly.
- `--init_A_Tmax`: maximum number of time bins used for data-driven `A`
  initialization.

The reference EM fit should be long enough to assign reliable states. Final
conditional `A,B` precision is controlled by the locked-M-step FDR bag jobs.

## FDR-Bags Stage (a)

First create the reference fit with `prism_EM_train3c.py`. In production runs
use an explicit tag, for example `--fitName <dataName>_jXXXX`, so this file
exists:

```bash
$basePath/prismFit/<dataName>_jXXXX.prismEM.npz
```

Then run `prism_FDR_Bags_train3c.py` once per bag. Each invocation:

1. Loads the EM-train output selected by `--emFitName` and the original spike
   file selected by `--dataName`.
2. Forms lag pairs `(S_hat[t], spikes[t-1], spikes[t])`.
3. Samples `--bag_frac` of those pairs without replacement.
4. Runs a locked M-step to refit only `A,B`.
5. Runs `--num_scrambles` per-neuron time-shuffle null refits using the same
   selected pair indices and locked labels.
6. Writes one self-contained bag file.

The output file is:

```bash
$basePath/prismFDR/<outFitName>.bag<bag_idx:03d>.prismFDRbag.npz
```

When the reference fit stem differs from the spike dataset stem, pass
`--emFitName <dataName>_emjXXXX`. This keeps input spikes at
`spikesData/<dataName>.spikes.npz` while reading the reference states from
`prismFit/<dataName>_emjXXXX.prismEM.npz`.

All bag indices, including `bag000`, are equivalent random pair subsets.

Example using 4 GPUs for bag 3:

```bash
basePath=/path/to/run
shortN=myDataset

time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_FDR_Bags_train3c.py \
  --basePath $basePath \
  --dataName $shortN \
  --emFitName ${shortN}_emj1234 \
  --outFitName ${shortN}_emj1234_rmfA \
  --bag_idx 3 \
  --bag_frac 0.8 \
  --epochs 240 \
  --num_scrambles 4 \
  --batch_size 4096
```

The bag driver loads locked-M-step defaults from the reference EM metadata,
but these can be overridden for FDR fitting:

- `--bag_idx`: integer bag index used in the output filename and RNG stream.
- `--emFitName`: EM-train input fit stem in `prismFit/`.
- `--outFitName`: Stage (a) output bag stem in `prismFDR/`.
  If omitted, it defaults to `<emFitName>_<hash4>` with a random
  four-character hex suffix.
- `--fdr_out_dir`: optional output directory; default is `$basePath/prismFDR`.
- `--bag_frac`: fraction of reference lag pairs selected without replacement.
- `--epochs`: locked M-step epochs for real and null refits.
- `--lr_mstep`, `--lr_end_factor`, `--lambda3`, `--rho_max`,
  `--prescale_m_step_4_ArhoMax`, `--batch_size`: locked-M-step overrides.
- `--init_A {data,ref,rand}` and `--init_B {data,ref,rand}`: real locked-fit
  initialization; default `data` to let each bag initialize from its sampled
  lag pairs.
- `--num_scrambles`: number of per-neuron time-shuffle null refits.
- `--min_roll_shift_sec`: exclusion zone around zero shift for each neuron;
  default `2`.
- `--null_A_init {scrambled_data,rand}`: fresh `A` initialization mode for
  each null refit; default `scrambled_data`.
- `--null_B_init {scrambled_data,rand}`: fresh `B` initialization mode for
  each null refit; default `scrambled_data`.

The bag trainer does not accept EM state-discovery controls such as
`--time_range_sec`, `--num_states`, `--num_em_iters`, `--pgd_iter`,
`--lr_estep`, `--lambda2`, `--init_states`, or `--decode_dwell_sec`.

## Running Multiple Bags

Stage (a) bags are independent jobs. For a small manual launch, vary
`--bag_idx`:

```bash
for bag in 0 1 2 3 4; do
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    ./prism_FDR_Bags_train3c.py \
    --basePath $basePath \
    --dataName $shortN \
    --emFitName ${shortN}_emj1234 \
    --outFitName ${shortN}_emj1234_rmfA \
    --bag_idx $bag \
    --bag_frac 0.8 \
    --epochs 240 \
    --num_scrambles 4
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
fdrFitName=${shortN}_emj1234_rmfA
agrFitName=${fdrFitName}_agrA
num_bags=5

./prism_EM_FDR_Bags_agregate3c.py \
  --basePath $basePath \
  --dataName $fdrFitName \
  --outAgrName $agrFitName \
  --num_bags $num_bags \
  --per_bag_quantile 0.99 \
  --stab_sel_thresh 0.7
```

The aggregator reads:

```bash
$basePath/prismFDR/<dataName>.bag000.prismFDRbag.npz
$basePath/prismFDR/<dataName>.bag001.prismFDRbag.npz
...
```

and writes:

```bash
$basePath/prismFit/<outAgrName>.prismEM.npz
```

If `--outAgrName` is omitted, the output stem defaults to `<dataName>_<hash4>`
with a random four-character hex suffix.

For example, with `fdrFitName=daleN100_2ba29b_c47b43_emj1234_rmfA` and
`outAgrName=${fdrFitName}_agrA`, the output stem is:

```bash
daleN100_2ba29b_c47b43_emj1234_rmfA_agrA
```

The output keeps the normal PRISM-EM fields expected by `prism_EM_eval3c.py`.
The connectivity fields are aggregated Stage (b) results, while the time
series fields (`S_hat`, `c_hat`, and `S_hat_CL`) come from the reference EM
fit used to lock the bags.

You can then run:

```bash
  ./prism_EM_eval3c.py \
  --basePath $basePath \
  --dataName $agrFitName \
  -p a e f g
```

To inspect one Stage (a) locked bag fit directly, pass the bag stem as
`--dataName`; names containing `bagNNN` are loaded from `prismFDR`:

```bash
./prism_EM_eval3c.py \
  --basePath $basePath \
  --dataName ${shortN}.bag000 \
  -p a e f g
```

Stage (b) also saves FDR diagnostics such as `selection_frequency`,
`selected_mask`, `A_mean_selected`, `A_sd_selected`, `A_mean_all`, `A_sd_all`,
`src_null_mean`, `src_null_sd`, `src_null_tau_mean`, `z_null`, and a compact
selected-edge table (`edge_i`, `edge_j`, `edge_sel_freq`, `edge_A_mean`,
`edge_A_sd_boot`, `edge_z_null`, `edge_src_null_mean`, `edge_src_null_sd`,
`edge_src_tau_mean`).

## Saved Bag Contents

Each `.prismFDRbag.npz` contains the real fit fields in the same style as
`prism_EM_train3c.py`, plus FDR-bagging fields including:

- `A_null`: null connectivity stack with shape `(num_scrambles, N, N)`.
- `B_null`: null bias stack with shape `(num_scrambles, num_states, N)`.
- `roll_shifts_bin`: realized circular shifts per scramble and neuron.
- `bag_pair_indices`: selected lag-pair row indices in the reference window.
- `bag_pair_dest_bins`: selected destination-bin indices in original time.
- `bag_pair_state`: locked state label for each selected pair.
- `null_*_epoch`: locked M-step histories for the null refits.

Metadata from the bagging workflow are grouped by stage. `bagsFDR_stageA`
contains pair-sampling settings, locked-M-step hyperparameters, null-refit
settings, per-scramble null initialization diagnostics, and reference-fit
provenance. The aggregate file also adds `bagsFDR_stageB` for the cross-bag
selection and output settings.
