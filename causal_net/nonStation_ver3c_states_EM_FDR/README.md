# PRISM-EM 3c Training

This directory contains the lag-1 PRISM-EM trainer, FDR bagging fitters, and
the Stage (c) de-biased refit.

The main pipeline entry points are:

- `prism_EM_train3c.py`: ordinary time-ordered EM fit used for state
  discovery.
- `prism_FDR_Bags_train3c.py`: one FDR-bagging Stage (a) job, i.e. one
  reference-locked A/B refit plus its per-neuron time-shuffle null refits.
- `prism_EM_FDR_Bags_aggregate3c.py`: EM-FDR-bagging Stage (b), i.e.
  aggregate all bag files into one eval-compatible fit.
- `prism_deBiasFit3c.py`: Stage (c), i.e. refit the Stage (b)-selected support
  without another sparsity penalty or pruning pass.

The reference and bag trainers share the core implementation in
`PrismEM_Workhorse3c.py`; Stage (c) uses `PrismDeBias_Workhorse3c.py`.

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

## Schema-versioned NPZ files

The core 3c pipeline uses `toolbox/Util_NumpyIOv2.py`. Every NPZ file written
through this module contains a required `schema.JSON` record with format
`Util_NumpyIO`, schema version `2`, and the encoding, NumPy dtype, and shape of
each payload record. Metadata are stored as strict JSON in `meta.JSON` and are
also described by the schema.

Use the v2 functions for all files passed through the schema-enabled pipeline:

```python
from toolbox.Util_NumpyIOv2 import (
    json_safe_metadata,
    read_data_npz,
    write_data_npz,
)

data, metadata = read_data_npz(input_file)
write_data_npz(data, output_file, metaD=json_safe_metadata(metadata))
```

The v2 reader deliberately uses `allow_pickle=False` and validates the entire
archive against `schema.JSON`. It rejects legacy schema-less files, mismatched
dtypes or shapes, undeclared records, and non-string object arrays. Payload
dictionaries and metadata must be JSON-compatible; NumPy scalars are converted
by `json_safe_metadata`, and non-finite floating-point metadata values become
JSON `null`. Do not write pipeline files directly with `numpy.savez*`, because
that bypasses the required schema.

The schema requirement is active in `gen_daleMatrices3c.py`,
`gen_nonStationarySpikes3c.py`, `view_daleMatrix3.py`,
`view_spikesTrain3.py`, `prep_bioexp3c.py`, `view_bioexp.py`,
`edgeMeterAccuracy3c.py`, `edgeMeterFidelity3c.py`, `prism_EM_train3c.py`,
`prism_FDR_Bags_train3c.py`, `prism_EM_FDR_Bags_aggregate3c.py`,
`prism_deBiasFit3c.py`, and `prism_EM_eval3c.py`. Consequently, the spike,
reference-fit, bag, aggregate, and de-biased files used together in a new run
must all be v2 archives.

## Shell Macros

The `.sh` files in this directory are convenience launch macros. They are
intended to be edited near the top before use, not treated as stable command
line interfaces.

- `fitPrismEM.sh`: thin wrapper around `prism_EM_train3c.py` for a single
  distributed EM fit. It checks that `--basePath`, `--dataName`, and
  `--num_states` were supplied, fixes a short default `--time_range_sec 0 80`,
  checks that four GPUs are visible, and launches `torchrun`. This is mostly
  useful for quick trainer smoke tests.

- `big_fit_bags.sh`: end-to-end example for one complete FDR/de-bias run on a
  chosen dataset and time range. It runs one reference EM fit, loops over
  `numBags` calls to `prism_FDR_Bags_train3c.py`, runs the Stage (b)
  aggregator, and then passes that aggregate directly to
  `prism_deBiasFit3c.py`. Edit `basePath`, `shortN`, `numStates`, `timeRange`,
  and the training/FDR hyperparameters at the top. Set `runAggregate=0` to
  skip aggregation, or `runDebias=0` to stop after aggregation; de-biasing
  requires aggregation to be enabled.

- `big_scanTime_FDR.sh`: end-to-end run for a single data-duration point in
  a scan. It takes three positional arguments:
  `./big_scanTime_FDR.sh <suffix> <start_min> <stop_min>`. The macro sets
  `timeRange=(start_min*60 stop_min*60)` seconds, runs reference EM, all FDR
  bags, and aggregation. The final aggregate name is
  `<shortN>_<suffix>_<start_min>to<stop_min>min`, which is convenient for
  later metric scans over duration or time window.

- `scan_FDR_BAG_hpar.sh`: cheap Stage (b)-only hyperparameter scan for an
  existing set of bag files. It does not rerun EM or bag fitting. It reuses
  `$basePath/prismFDR/${fdrBagsName}.bagNNN.prismFDRbag.npz` and sweeps
  `stabSelScan` at fixed `base_per_bag_quantile`, then sweeps
  `perBagQuantileScan` at fixed `base_stab_sel_thresh`. It writes aggregate
  files named from `outAgrName` and prints ready-to-run `edgeMaterAbs3c.py`
  metric commands for the two scans.

- `docs/buildTex.sh`: numbered LaTeX builder for the documentation sources.
  It places intermediate build products under `docs/tmp/` and leaves only
  `buildTex.sh`, `.tex`, and `.pdf` files at the top level of `docs/`.

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

1. Loads the EM-train output selected by `--emFitName`.
2. Recovers the original spike-file stem from that EM fit's provenance
   metadata.
3. Forms lag pairs `(S_hat[t], spikes[t-1], spikes[t])`.
4. Samples `--bag_frac` of those pairs without replacement.
5. Runs a locked M-step to refit only `A,B`.
6. Runs `--num_scrambles` per-neuron time-shuffle null refits using the same
   selected pair indices and locked labels.
7. Writes one self-contained bag file.

The output file is:

```bash
$basePath/prismFDR/<outFitName>.bag<bag_idx:03d>.prismFDRbag.npz
```

The bag trainer does not accept `--dataName`. This is intentional: the source
spike file is derived from the reference EM metadata, which prevents mixing
states from one dataset with spikes from another.

All bag indices, including `bag000`, are equivalent random pair subsets.

Example using 4 GPUs for bag 3:

```bash
basePath=/path/to/run
shortN=myDataset

time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_FDR_Bags_train3c.py \
  --basePath $basePath \
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
fdrBagsName=${shortN}_emj1234_rmfA
agrFitName=${fdrBagsName}_agrA
num_bags=5

./prism_EM_FDR_Bags_aggregate3c.py \
  --basePath $basePath \
  --dataName $fdrBagsName \
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

For example, with `fdrBagsName=daleN100_2ba29b_c47b43_emj1234_rmfA` and
`outAgrName=${fdrBagsName}_agrA`, the output stem is:

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

## De-biased Fit Stage (c)

Stage (b) selects the connectivity support, but its edge amplitudes still come
from regularized bag fits. `prism_deBiasFit3c.py` removes that shrinkage by
refitting `A` and `B` on the Stage (b) support. All selected off-diagonal edges
and all diagonal entries are active during this refit; unselected off-diagonal
entries remain zero. Dale signs are enforced and there is no second pruning or
selection pass.

To run Stage (c) manually on an aggregate:

```bash
basePath=/path/to/run
fdrAgrName=myDataset_em1234_fdr1234_agr1234
debiasFitName=${fdrAgrName}_debias

time torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_deBiasFit3c.py \
  --basePath "$basePath" \
  --fdrFitName "$fdrAgrName" \
  --outFitName "$debiasFitName" \
  --state_mode locked \
  --m_epochs 180 \
  --batch_size 4096
```

This reads:

```bash
$basePath/prismFit/<fdrAgrName>.prismEM.npz
```

and writes:

```bash
$basePath/prismFit/<debiasFitName>.prismEM.npz
```

`state_mode=locked` preserves the Stage (b) state assignments and performs one
fixed-state M-step. `state_mode=refit` alternates E- and M-steps; use
`--num_debias_iters` to control their number. The default `state_mode=auto`
chooses `refit` for multi-state inputs and `locked` for a single state.

The Stage (c) archive preserves Stage (b) results such as `A_hat`, `B_hat`,
and `selected_mask`, while adding the primary de-biased results as `A_debias`
and `B_debias`. `prism_EM_eval3c.py` recognizes Stage (c) metadata and displays
those de-biased arrays:

```bash
./prism_EM_eval3c.py \
  --basePath "$basePath" \
  --dataName "$debiasFitName" \
  -p a b c e h i
```

`big_fit_bags.sh` now includes this step immediately after Stage (b). Its
default settings tie the de-bias epoch count and batch size to the bag-fit
settings:

```bash
runAggregate=1
runDebias=1
debiasStateMode=locked
debiasEpochs=$bagEpochs
debiasBatchSize=$bagBatchSize
```

For each new random run tag, the macro creates an aggregate named
`${fdrAgrName}.prismEM.npz`, uses `fdrAgrName` as `--fdrFitName`, and writes
`${fdrAgrName}_debias.prismEM.npz`. Both paths are printed when the pipeline
finishes.

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
