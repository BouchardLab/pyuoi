#!/usr/bin/env python3
"""
Single-GPU prism M-step training for non-stationary Poisson GLM.

Fits a shared connectivity matrix A and per-state bias vectors B_hat[m]
using ground-truth hard state assignments S_true(t) from prismTruth.
"""

import os
import time
import secrets
import argparse
import random

import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from UtilTorch import check_gpu_availability


class SwitchingPoissonGLModel(nn.Module):
    """Shared A, per-state B[m], hard assignment by state index."""
    def __init__(self, n_neurons, n_states, eta_clip):
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_states = int(n_states)
        self.eta_clip = float(eta_clip)
        self.A = nn.Parameter(torch.randn(self.n_neurons, self.n_neurons) * 0.1)
        self.B = nn.Parameter(torch.randn(self.n_states, self.n_neurons) * 0.1)

    def forward(self, y_prev, s_idx, dt):
        linear = torch.addmm(self.B[s_idx], y_prev, self.A.t())
        linear = torch.clamp(linear, max=self.eta_clip)
        return torch.exp(linear) * dt


class NumpyTripletDataset(Dataset):
    def __init__(self, x_np, y_np, s_np):
        assert x_np.shape[0] == y_np.shape[0] == s_np.shape[0]
        self.x = x_np
        self.y = y_np
        self.s = s_np

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]).to(dtype=torch.float32),
            torch.from_numpy(self.y[idx]).to(dtype=torch.float32),
            torch.tensor(self.s[idx], dtype=torch.long),
        )


def make_loader_xys(x_np, y_np, s_np, batch_size, shuffle=True):
    ds = NumpyTripletDataset(x_np, y_np, s_np)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=8,
    )


def poisson_nll_weighted(spikes, targets, firing_rates):
    eps = 1e-8
    firing_rates_safe = torch.maximum(firing_rates, torch.tensor(0.1, device=firing_rates.device))
    weights = 1.0 / firing_rates_safe
    weights = weights / torch.mean(weights)
    weights = weights.unsqueeze(0)
    loss = -weights * targets * torch.log(spikes + eps) + weights * spikes
    return loss.mean()


def poisson_nll_weighted_per_sample(spikes, targets, firing_rates):
    eps = 1e-8
    firing_rates_safe = torch.maximum(firing_rates, torch.tensor(0.1, device=firing_rates.device))
    weights = 1.0 / firing_rates_safe
    weights = weights / torch.mean(weights)
    weights = weights.unsqueeze(0)
    loss = -weights * targets * torch.log(spikes + eps) + weights * spikes
    return loss.mean(dim=1)


def offdiag_soft_threshold_(A, lr, lam):
    if lam <= 0.0:
        return
    with torch.no_grad():
        n = A.shape[0]
        off_diag = ~torch.eye(n, dtype=torch.bool, device=A.device)
        t = lr * lam
        A_off = A[off_diag]
        A[off_diag] = A_off.sign() * (A_off.abs() - t).clamp(min=0.0)


def enforce_spectral_radius_(A, rho_max, eps=1e-12):
    if rho_max is None:
        return
    if rho_max <= 0.0:
        raise ValueError(f"rho_max must be > 0, got {rho_max}")
    with torch.no_grad():
        rho = torch.linalg.eigvals(A).abs().max()
        if torch.isfinite(rho) and (rho > rho_max):
            A.mul_(float(rho_max) / float(rho + eps))


def preprocess_data_with_states(spikes, s_true, args):
    y = np.asarray(spikes)
    s_true = np.asarray(s_true).astype(np.int64)
    nt, nn = y.shape
    assert s_true.shape[0] == nt, "S_true length mismatch with spikes length"

    if args.desyncTime:
        seed = int(time.time() * 1000) % 1000000
        np.random.seed(seed)
        shift_amounts = np.random.randint(1, nt // 4, size=nn, dtype=np.int32)
        y_shifted = np.zeros_like(y)
        for neuron_idx in range(nn):
            shift_amount = int(shift_amounts[neuron_idx])
            y_shifted[:, neuron_idx] = np.roll(y[:, neuron_idx], shift_amount)
        y = y_shifted
        print(f"Applied time decorrelation shifts (seed={seed})")

    max_pairs = nt - 1
    num_samples = args.num_samples
    if num_samples is None or num_samples > max_pairs:
        num_samples = max_pairs
    x_np = y[:num_samples]
    yt_np = y[1:num_samples + 1]
    s_np = s_true[1:num_samples + 1]

    if args.dropDataFrac > 0:
        n_pairs = x_np.shape[0]
        drop_seed = (int(time.time() * 1000) + np.random.randint(0, 1000)) % (2**32)
        np.random.seed(drop_seed)
        random.seed(drop_seed)
        n_keep = int(n_pairs * (1.0 - args.dropDataFrac))
        keep_indices = np.random.choice(n_pairs, size=n_keep, replace=False)
        keep_indices = np.sort(keep_indices)
        x_np = x_np[keep_indices]
        yt_np = yt_np[keep_indices]
        s_np = s_np[keep_indices]
        print(f"Dropped {args.dropDataFrac:.1%} of data, keeping {x_np.shape[0]} samples (seed={drop_seed})")

    return x_np.astype(np.float32), yt_np.astype(np.float32), s_np.astype(np.int64)


def train_switching_mstep_model(
    model,
    device,
    train_loader,
    n_epochs,
    lr,
    dt,
    firing_rates,
    L1_alpha=0.0,
    use_scheduler=True,
    lr_end_factor=0.03,
    rho_max=0.97,
    rho_enforce_every_batch=20,
    L1_prune_epoch=0,
    minW=1e-6,
):
    if L1_prune_epoch < 0:
        raise ValueError(f"L1_prune_epoch must be >= 0, got {L1_prune_epoch}")
    if rho_enforce_every_batch < 1:
        raise ValueError(f"rho_enforce_every_batch must be >= 1, got {rho_enforce_every_batch}")
    use_fused = isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available()
    assert use_fused, "This trainer expects CUDA fused Adam"

    optimizer = optim.Adam(model.parameters(), lr=lr, fused=True)
    scheduler = (
        optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=n_epochs
        )
        if use_scheduler
        else None
    )

    n = model.n_neurons
    diag_mask = torch.eye(n, device=device, dtype=torch.bool)
    off_diag_mask = ~diag_mask
    l1_weight_matrix = torch.ones(n, n, device=device)
    l1_weight_matrix[diag_mask] = 0.0
    firing_rates_t = torch.tensor(firing_rates, dtype=torch.float32, device=device)

    loss_epoch, loss_nll_epoch, loss_l1_epoch = [], [], []
    learning_rates, rho_epoch, nz_edges_epoch, sparsity_epoch = [], [], [], []
    loss_state_total_epoch = []

    t_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        apply_prune = epoch >= L1_prune_epoch
        sum_tot = 0.0
        sum_nll = 0.0
        sum_l1 = 0.0
        state_nll_sum = np.zeros((model.n_states,), dtype=np.float64)
        state_n_count = np.zeros((model.n_states,), dtype=np.int64)

        for batch_idx, (y_prev, y_curr, s_idx) in enumerate(train_loader):
            y_prev = y_prev.float().to(device, non_blocking=True)
            y_curr = y_curr.float().to(device, non_blocking=True)
            s_idx = s_idx.long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            spikes = model(y_prev, s_idx, dt=dt)
            nll_per_sample = poisson_nll_weighted_per_sample(spikes, y_curr, firing_rates_t)
            base_loss = nll_per_sample.mean()
            if L1_alpha > 0:
                l1_term = L1_alpha * torch.mean(torch.abs(model.A) * l1_weight_matrix)
            else:
                l1_term = torch.tensor(0.0, device=device)
            total_loss = base_loss + l1_term
            total_loss.backward()
            optimizer.step()

            if apply_prune and (L1_alpha > 0):
                offdiag_soft_threshold_(model.A, optimizer.param_groups[0]["lr"], L1_alpha)
            if batch_idx % rho_enforce_every_batch == 0:
                enforce_spectral_radius_(model.A, rho_max)

            sum_tot += float(total_loss.item())
            sum_nll += float(base_loss.item())
            sum_l1 += float(l1_term.item())
            for m in range(model.n_states):
                mask = (s_idx == m)
                cnt = int(mask.sum().item())
                if cnt == 0:
                    continue
                state_nll_sum[m] += float(nll_per_sample[mask].sum().item())
                state_n_count[m] += cnt

        n_batches = float(len(train_loader))
        loss_epoch.append(sum_tot / n_batches)
        loss_nll_epoch.append(sum_nll / n_batches)
        loss_l1_epoch.append(sum_l1 / n_batches)
        state_nll_epoch = state_nll_sum / np.maximum(1, state_n_count)
        state_total_epoch = state_nll_epoch + loss_l1_epoch[-1]
        loss_state_total_epoch.append(state_total_epoch)
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))

        with torch.no_grad():
            a_off = model.A[off_diag_mask]
            nz_off = int((a_off.abs() > minW).sum().item())
            n_off = int(off_diag_mask.sum().item())
            sparsity = 1.0 - nz_off / max(1, n_off)
            rho = float(torch.linalg.eigvals(model.A).abs().max().item())
        nz_edges_epoch.append(nz_off)
        sparsity_epoch.append(float(sparsity))
        rho_epoch.append(rho)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 20 == 0 or (epoch + 1) == n_epochs:
            print(
                f"Epoch {epoch+1}/{n_epochs}:  NLL={loss_nll_epoch[-1]:.5g}, "
                f"L1={loss_l1_epoch[-1]:.5g}, A_sparsity={sparsity:.3f}, "
                f"nz_offdiag={nz_off}, rho(A)={rho:.4f}, Elapsed={(time.time() - t_start):.1f}s"
            )

    return {
        "loss_epoch": np.asarray(loss_epoch, dtype=np.float64),
        "loss_nll_epoch": np.asarray(loss_nll_epoch, dtype=np.float64),
        "loss_l1_epoch": np.asarray(loss_l1_epoch, dtype=np.float64),
        "loss_state_total_epoch": np.asarray(loss_state_total_epoch, dtype=np.float64),
        "learning_rates": np.asarray(learning_rates, dtype=np.float64),
        "rho_epoch": np.asarray(rho_epoch, dtype=np.float64),
        "nz_edges_epoch": np.asarray(nz_edges_epoch, dtype=np.int64),
        "sparsity_epoch": np.asarray(sparsity_epoch, dtype=np.float64),
    }


#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataName", type=str, default="dale_2aee70")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input/output data")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L1_alpha", type=float, default=0.02, help="L1 regularization strength, higher=more sparse (0: disable soft-thresholding)")
    parser.add_argument("--rho_max", type=float, default=0.97, help="Maximum allowed spectral radius of A; projection applied after each batch")
    parser.add_argument("--rho_enforce_every_batch", type=int, default=20, help="Apply spectral-radius projection every N batches")
    parser.add_argument("--L1_prune_epoch", type=int, default=50, help="Delay L1 edge-pruning only; spectral-radius correction starts immediately")
    parser.add_argument("--minW", type=float, default=0.01, help="Threshold for A-matrix eval, not for fitting")
    parser.add_argument("--fitName", type=str, default=None)
    parser.add_argument("--desyncTime", action='store_true', help="If true completely shuffle time axis for input data, independently for all channels")
    parser.add_argument("--dropDataFrac", type=float, default=0.0, help="Fraction of training samples to randomly drop (0.0=use all data, 0.3=drop 30%%)")
    parser.add_argument("--verb", "-v", type=int, default=1, help="Verbosity level")

    args = parser.parse_args()

    inpPath = os.path.join(args.basePath, 'spikesData')
    outPath = os.path.join(args.basePath, 'prismFit')
    os.makedirs(outPath, exist_ok=True)

    device = check_gpu_availability()
    print("\nPrism M-step Config:", vars(args), "\n")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    gpu_name = torch.cuda.get_device_name(device) if isinstance(device, torch.device) and device.type == 'cuda' else str(device)
    print("Using device %s : %s" % (str(device), gpu_name))

    spikesFF = os.path.join(inpPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=True)
    if args.verb > 1:
        pprint(spikeMD)
    dataYield = spikeD['spikes']
    dataRates = spikeD['single_rates']
    step_size = float(spikeMD['time_step_sec'])
    eta_clip = float(spikeMD["poisson_eta_clip"])
    prov = spikeMD["provenance"]

    assert spikeMD['data_type'] == 'simPrism'
    _, Nn = dataYield.shape

    truthF = prov["state_transition_file"]
    truthFF = os.path.join(inpPath, f"{truthF}.prismTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nprismTruth metadata:")
        pprint(trueMD)
    S_true = trueD["S_true"].astype(np.int64)
    evol_conf = trueMD.get("evol_conf", {})
    if "num_states" in evol_conf:
        n_states = int(evol_conf["num_states"])
    elif "B_true" in trueD:
        n_states = int(np.asarray(trueD["B_true"]).shape[0])
    else:
        n_states = int(np.max(S_true) + 1)

    x_np, yt_np, s_np = preprocess_data_with_states(dataYield, S_true, args)
    n_pairs = x_np.shape[0]
    print(f"Preprocessed data: X={x_np.shape}, Y={yt_np.shape}, S={s_np.shape}, n_pairs={n_pairs}")

    assert n_pairs >= args.batch_size, (
        f"ERROR: Not enough samples ({n_pairs}) for batch size ({args.batch_size}) after data dropping."
    )

    if np.any(s_np < 0) or np.any(s_np >= n_states):
        raise ValueError(f"S_true contains state id outside [0,{n_states-1}]")

    state_counts = np.bincount(s_np, minlength=n_states)
    if args.verb > 0:
        frac = state_counts / max(1, state_counts.sum())
        msg = "  ".join(f"s{m}:{state_counts[m]}({frac[m]:.1%})" for m in range(n_states))
        print(f"Training state coverage: {msg}")

    train_loader = make_loader_xys(x_np, yt_np, s_np, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded pairs={n_pairs/1000:.3f}k, Nn={Nn}, M={n_states}, batch_size={args.batch_size}")

    model = SwitchingPoissonGLModel(Nn, n_states=n_states, eta_clip=eta_clip).to(device)
    start_time = time.time()
    train_hist = train_switching_mstep_model(
        model=model,
        device=device,
        train_loader=train_loader,
        n_epochs=args.num_epochs,
        lr=args.lr,
        dt=step_size,
        firing_rates=dataRates,
        L1_alpha=args.L1_alpha,
        use_scheduler=True,
        rho_max=args.rho_max,
        rho_enforce_every_batch=args.rho_enforce_every_batch,
        L1_prune_epoch=args.L1_prune_epoch,
        minW=args.minW,
    )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")

    if args.fitName is None:
        hash6 = secrets.token_hex(3)
        fit_core = f"{args.dataName}-Mstep-{hash6}"
    else:
        fit_core = args.fitName

    A_shared = model.A.detach().cpu().numpy()
    B_hat = model.B.detach().cpu().numpy()
    A_hat = A_shared
    E_hat = (np.abs(A_hat) > 1e-5).astype(np.uint8)

    outD = {
        "A_hat": A_hat,
        "A_shared": A_shared,
        "B_hat": B_hat,
        "E_hat": E_hat,
        "single_rates": dataRates,
        **train_hist,
    }

    outMD = dict(spikeMD)
    outMD["fit_type"] = "prismMstep"
    outMD["train"] = {
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "num_samples_used": int(n_pairs),
        "lr": float(args.lr),
        "learning_rate": float(args.lr),
        "L1_alpha": float(args.L1_alpha),
        "lambda3": float(args.L1_alpha),
        "rho_max": float(args.rho_max),
        "rho_enforce_every_batch": int(args.rho_enforce_every_batch),
        "L1_prune_epoch": int(args.L1_prune_epoch),
        "minW": float(args.minW),
        "dropDataFrac": float(args.dropDataFrac),
        "time_step_sec": float(step_size),
        "eta_clip": float(eta_clip),
        "num_neurons": int(Nn),
        "num_states": int(n_states),
        "training_time_sec": float(total_time),
    }
    outMD["evol_conf"] = trueMD.get("evol_conf", {})
    outMD["dale_conf"] = trueMD.get("dale_conf", {})
    outMD["provenance"] = dict(spikeMD.get("provenance", {}))
    outMD["provenance"]["output_mstep_file"] = fit_core

    if args.verb > 1:
        pprint(outMD)
    fitFF = os.path.join(outPath, f"{fit_core}.prismMstep.npz")
    write_data_npz(outD, fitFF, metaD=outMD)

    print('    basePath=' + args.basePath)
    print('  ./prism_Mstep_eval.py --basePath $basePath  --dataName %s  -p a b c d  \n ' % (fit_core,))


if __name__ == "__main__":
    main()
