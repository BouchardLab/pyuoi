# User Guide: Experimental Data Processing Pipeline (`nonStation_ver3c_states_EM_FDR`)

This document describes the workflow for processing raw microelectrode array (MEA) experimental recordings into causal network reconstructions using the **PRISM-EM + FDR Bagging** framework in `nonStation_ver3c_states_EM_FDR`.

---

## 1. Overview & Computational Environment

The pipeline extracts directed causal network connectivity from multi-channel spike trains under non-stationary network states. The workflow is optimized for execution on **NERSC Perlmutter (PM)** GPU nodes:
- **Recommended Hardware**: 1 GPU node with 4× NVIDIA A100 GPUs.
- **Environment Setup**: On Perlmutter, it suffices to run:
  ```bash
  module load pytorch
  ```
  This single command loads PyTorch, NumPy, Pandas, Matplotlib, and all required scientific Python libraries for the entire pipeline. No additional conda environment or package installation is needed.
- **Thread Settings**: Restrict CPU thread oversaturation before running:
  ```bash
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  ```
- **Multi-GPU Runtime**: PyTorch Distributed Data Parallel (`torchrun --standalone --nnodes=1 --nproc_per_node=4`).

---

## 2. Storage Strategy & Directory Hierarchy

To ensure high performance during model training while preventing filesystem clutter, data storage is separated into two tiers:

```
[ CFS: Long-Term Archive ]
  └── /global/cfs/cdirs/m2043/causal_inference/
        └── <sessionName>/ (Raw chip recordings, spike sorting outputs, metadata)
               │
               ▼  prep_bioexp3c.py (Format conversion & temporal binning)
[ SCRATCH: Working Directory ]
  └── /pscratch/sd/b/<user>/<project>/
        ├── spikesData/   (Preprocessed .spikes.npz & .bioExp.npz)
        ├── prismFit/     (Reference EM, aggregated, and de-biased model fits)
        ├── prismFDR/     (Per-bag FDR model checkpoints)
        └── plots/        (Multi-page diagnostic and evaluation figures)
```

### Raw Experimental Data (CFS)
Raw recording and spike-sorting outputs reside on the Community File System (CFS):
```bash
/global/cfs/cdirs/m2043/causal_inference/
```

Directory paths within CFS follow a descriptive, hierarchical structure:
```
<cell_line>/<date_yymmdd>/<chip_id>/<assay_type>/<run_number>/<well_number>
```
*Example session path:*
```bash
/global/cfs/cdirs/m2043/causal_inference/KCL_experiment/260729/M07420/Network/000018/well005
```

Each raw session directory contains:
- `spike_times.npy`: Spike timestamps per sorted unit.
- `raw_mean_templates.npy`: Mean waveform templates per unit across MEA channels.
- `metrics_curated.xlsx` (or `quality_metrics.xlsx`): Unit quality metrics and 2D MEA coordinates (`loc_x`, `loc_y`).

### Working Data (SCRATCH)
Intermediate data, model checkpoints, and training outputs are placed on Perlmutter **SCRATCH** (`/pscratch/sd/...`):
- High I/O throughput for multi-GPU training.
- Automatic retention policy (SCRATCH autocleans older files), preventing long-term clutter on project CFS quota.

---

## 3. Step 1: Raw Data Ingestion & Preprocessing (`prep_bioexp3c.py`)

The first step converts hardware-specific recording files into standardized, NumPy-based (`.npz`, schema v2) archives optimized for causal network fitting.

### Conversion Script Example

```bash
# 1. Define CFS source and session
expPath=/global/cfs/cdirs/m2043/causal_inference/
sesN=KCL_experiment/260729/M07420/Network/000018/well005

# 2. Define an informative short name
shortN=MouseKCL_260729_w5_r18_1hz

# 3. Define scratch working directory
basePath=/pscratch/sd/b/balewski/2026_causalNet_Aug23
dataPath=${basePath}/spikesData/
mkdir -p "$dataPath"

# 4. Run preprocessing
./prep_bioexp3c.py \
  --sessionName "$sesN" \
  --dataPath "$dataPath" \
  --expPath "$expPath" \
  --freqRange 1 50 \
  --shortName "$shortN"
```

### Key Principles for Preprocessing

1. **Informative Short Name (`--shortName`)**:
   - The output file name must retain key identifiers of the raw recording:
     - Sample / cell type (e.g., `MouseKCL`)
     - Recording date (`260729`)
     - Well number (`w5`)
     - Run number (`r18`)
     - Low-frequency filter cut (`1hz`)
   - **Why this matters**: This short name propagates through model files, evaluation logs, and plot titles. It must remain unique, descriptive, and sufficiently compact to fit on multi-panel figure headers.

2. **Always Convert the Full Run**:
   - `prep_bioexp3c.py` always processes the **entire recording duration**.
   - Do **not** pre-slice data during ingestion; temporal clipping (e.g., analyzing 0–1h vs. 2–4h) is handled downstream by training and evaluation scripts (`--time_range_sec`).

3. **Dual Output Files**:
   `prep_bioexp3c.py` writes two paired NPZ files into `${dataPath}/`:
   - `<shortName>.spikes.npz`:
     Contains the binned spike-count matrix ($T \times N$) used directly by the GLM fitters.
   - `<shortName>.bioExp.npz`:
     Contains all biological and hardware metadata: unit mapping, MEA 2D coordinates (`node_positions`), firing rate distributions, waveform templates, and curated metrics.
   - **Important**: Both files are required for meaningful evaluation and visualization of network topology and physical wiring.

---

## 4. Quality Control & Exploratory Visualization

After data ingestion, inspect recording features and spike trains before launching expensive GPU training jobs.

### Visualizing Experimental Features & MEA Layout (`view_bioexp.py`)
Inspect firing rate distributions, temporal stability, 2D electrode placement, and metric correlations:
```bash
./view_bioexp.py \
  --dataPath "$dataPath" \
  --dataName "$shortN" \
  -p a b c d e \
  -X
```
- Available plot pages:
  - `a`: Firing rate histogram across all accepted units.
  - `b`: Population firing rate vs. time.
  - `c`: Physical 2D MEA channel layout with neuron coordinates.
  - `d`: Quality metrics distributions (amplitude, SNR, isolation).
  - `e`: Metric correlation matrix.

### Visualizing Spike Trains & Bursts (`view_spikesTrain3.py`)
Inspect raw spike trains over a specific time window:
```bash
./view_spikesTrain3.py \
  --basePath "$basePath" \
  --dataName "$shortN" \
  -m -1 \
  -T 0 60 \
  -p a b \
  -X
```
*(Setting `-m -1` instructs the script to read directly from `spikesData/`).*

---

## 5. Network Reconstruction Pipeline

The core reconstruction pipeline consists of four sequential stages:

```
                  ┌──────────────────────────────────────────────┐
                  │ 1. Reference EM Fit: State Discovery        │
                  │    prism_EM_train3c.py                       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 2. FDR Bagging: Locked M-Step Refits         │
                  │    prism_FDR_Bags_train3c.py (bags 0 .. 10)  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 3. Bag Aggregation & Stability Selection    │
                  │    prism_EM_FDR_Bags_aggregate3c.py          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 4. De-biased Support Refit                   │
                  │    prism_deBiasFit3c.py                      │
                  └──────────────────────────────────────────────┘
```

### Stage 1: Reference EM Fit (`prism_EM_train3c.py`)
Discovers discrete network states (e.g. baseline vs. burst state) and fits initial interaction matrices:
```bash
emFitName="${shortN}_0to1h_emReference"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_EM_train3c.py \
  --basePath "$basePath" \
  --dataName "$shortN" \
  --fitName "$emFitName" \
  --time_range_sec 0 3600 \
  --num_states 2 \
  --decode_dwell_sec 0.10 \
  --num_em_iters 12 \
  --m_epochs 2 \
  --batch_size 4096 \
  --delay_em_iter_4_lrDecay 4 12 \
  --delay_em_iter_4_ArhoMax 6 12 \
  --delay_em_iter_4_Aprune 6
```
- **Output**: `${basePath}/prismFit/${emFitName}.prismEM.npz`

### Stage 2: FDR Bagging (`prism_FDR_Bags_train3c.py`)
Runs multiple training repetitions (typically 10–11 bags). Each bag fits a sub-sampled partition of the dataset (`--bag_frac 0.8`) and trains scrambled null models to calibrate false discovery rates:
```bash
fdrBagsName="${emFitName}_fdr"
numBags=11

for ((bag=0; bag<numBags; bag++)); do
    echo "Running bag $bag / $((numBags - 1))..."
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
      ./prism_FDR_Bags_train3c.py \
      --basePath "$basePath" \
      --emFitName "$emFitName" \
      --outFitName "$fdrBagsName" \
      --bag_idx "$bag" \
      --bag_frac 0.8 \
      --epochs 180 \
      --batch_size 4096 \
      --num_scrambles 6
done
```
- **Output**: `${basePath}/prismFDR/${fdrBagsName}.bag000.prismFDRbag.npz` ... `.bag010...`

### Stage 3: Bag Aggregation (`prism_EM_FDR_Bags_aggregate3c.py`)
Aggregates signal and null distributions across all bags using quantile thresholding and stability selection:
```bash
fdrAgrName="${fdrBagsName}_agr"

./prism_EM_FDR_Bags_aggregate3c.py \
  --basePath "$basePath" \
  --dataName "$fdrBagsName" \
  --outAgrName "$fdrAgrName" \
  --num_bags "$numBags" \
  --per_bag_quantile 0.97 \
  --stab_sel_thresh 0.7
```
- **Output**: `${basePath}/prismFit/${fdrAgrName}.prismEM.npz`

### Stage 4: De-biased Support Refitting (`prism_deBiasFit3c.py`)
Refits coupling coefficients on the aggregate network support without sparsity penalties to eliminate regularization shrinkage bias:
```bash
debiasFitName="${fdrAgrName}_deb"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ./prism_deBiasFit3c.py \
  --basePath "$basePath" \
  --fdrFitName "$fdrAgrName" \
  --outFitName "$debiasFitName" \
  --state_mode locked \
  --decode_dwell_sec 0.10 \
  --m_epochs 180 \
  --batch_size 4096
```
- **Output**: `${basePath}/prismFit/${debiasFitName}.prismEM.npz`

---

## 6. Evaluation & Multi-Page Diagnostics (`prism_EM_eval3c.py`)

After each fitting stage (EM reference, aggregated FDR, or de-biased fit), run `prism_EM_eval3c.py` to generate comprehensive multi-page diagnostic plots (typically 5–10 pages).

### Evaluation Command Example
```bash
./prism_EM_eval3c.py \
  --basePath "$basePath" \
  --dataName "$debiasFitName" \
  -p a b c d e f g h i j k \
  --plotFormat png \
  -X
```

### Plot Page Reference

| Plot Flag | Title / Description | Purpose |
|:---:|:---|:---|
| **`a`** | Fit Summary | Log-likelihood optimization curves, iteration loss, and global edge counts |
| **`b`** | Fitted $A$ Matrix | Heatmap of directed causal interactions and off-diagonal histograms |
| **`c`** | Inferred State Sequence | Latent state assignment timeline, state probabilities, and dwell times |
| **`d`** | Population Bursts | Multi-neuron firing raster with detected population burst windows |
| **`e`** | Outgoing Edge Degree | Node outgoing degree ($k_{out}$) vs. baseline neuron firing rates |
| **`f`** | 2D Spatial Connectivity | Physical MEA wiring diagram overlaying directed excitatory/inhibitory edges |
| **`g`** | FDR Bag Selection | Bag-by-bag acceptance probabilities and stability selection thresholding |
| **`h`** | Weight Distributions | Excitatory vs. inhibitory interaction weight distributions |
| **`i`** | Off-Diagonal Investigation | Detailed distribution of non-zero coupling coefficients |
| **`j`** | Excitatory Waveforms | Mean action potential waveforms for top excitatory hub neurons |
| **`k`** | Inhibitory Waveforms | Mean action potential waveforms for top inhibitory hub neurons |

Plots are saved to `${basePath}/plots/`.

---

## 7. End-to-End Execution Script (`big_fit_bags.sh`) & Batch Submission

The script `big_fit_bags.sh` encapsulates the complete reference EM $\to$ FDR bagging $\to$ aggregation $\to$ de-bias pipeline.

### Interactive Execution (1 Node, 4 GPUs)
For shorter windows or test runs (up to 4 hours in the interactive queue):
```bash
# Request Perlmutter interactive GPU node
salloc -q interactive -C gpu -t 4:00:00 -N 1 -A m2043

# Run pipeline on a designated time window
./big_fit_bags.sh 0to1h
```
Supported time window options: `0to1h`, `1to2h`, `2to3h`, `3to4h`, `0to2h`, `2to4h`, `0to4h`.

### Production Batch Submission (15h+ Long Jobs with `batchFitBags.slr`)
For full multi-hour recordings (e.g., `0to4h`), fitting 11 FDR bags with time-scrambled nulls requires substantial wall clock time (~12–15+ hours). 

The SLURM job script `batchFitBags.slr` demonstrates how to launch long-running jobs in Perlmutter's `regular` queue:

```bash
#!/bin/bash
# Submit batch job (or SLURM array across multiple recordings):
#   sbatch batchFitBags.slr

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -A m2043
#SBATCH --time=17:48:00   -q regular
#SBATCH --output=outj/%A_%a.out
#SBATCH --licenses=scratch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p outj

echo "S: job=${SLURM_JOB_ID:-local} host=$(hostname)"

# Executes full end-to-end pipeline on the 4-hour window
time ./big_fit_bags.sh 0to4h
```

Submit the job using:
```bash
sbatch batchFitBags.slr
```

Job progress and terminal outputs are logged automatically under `outj/`.

---

## 8. Summary Checklist for Experimental Runs

1. [ ] Load PyTorch module on PM: `module load pytorch`
2. [ ] Raw data located on CFS: `/global/cfs/cdirs/m2043/causal_inference/<sessionName>`
3. [ ] Choose a descriptive, unique short name (e.g. `MouseKCL_260729_w5_r18_1hz`).
4. [ ] Run `prep_bioexp3c.py` to write paired `.spikes.npz` and `.bioExp.npz` to SCRATCH.
5. [ ] Run `view_bioexp.py` to verify unit quality metrics and 2D spatial layout.
6. [ ] Launch pipeline via `big_fit_bags.sh` (interactive) or submit long 15h+ runs with `sbatch batchFitBags.slr`.
7. [ ] Run `prism_EM_eval3c.py` to produce diagnostic figures and inspect reconstructed connectivity.
