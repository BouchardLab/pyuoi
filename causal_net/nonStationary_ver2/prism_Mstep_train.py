#!/usr/bin/env python3
"""
Partial M-step training for non-stationary Poisson dLDS with hard state assignment.
Loads FDR-selected initialization, spike data, and prismTruth state sequence,
then updates fA/fB with Poisson deviance + group lasso and spectral projection.
"""

import os
import time
import math
import argparse
from pprint import pprint
import secrets

import numpy as np
import torch

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from UtilTorch import check_gpu_availability


def parse_args():
    parser = argparse.ArgumentParser(description="Partial M-step training for prism model (hard state assignment)")
    parser.add_argument("--fdrName", type=str, required=True, help="Base name of FDRselected file in lassoFdrFit/")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2026_causalNet_tmp2/", help="Head dir for input/output data")
    parser.add_argument("--num_epochs", type=int, default=251)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--end_lr_frac", type=float, default=0.2, help="Final LR as fraction of initial LR (linear decay). Default 0.2 = 1/5.")
    parser.add_argument("--lambda3", type=float, default=1.0, help="Group lasso strength")
    parser.add_argument("--epsW", type=float, default=0.02, help="Group-norm floor for zeroing weights")
    parser.add_argument("--minW", type=float, default=0.02, help="Edge magnitude threshold for reporting counts")
    parser.add_argument("--chunk_size", type=int, default=128*1024, help="Time batching size")
    parser.add_argument("--rndStart", action="store_true", help="Randomly initialize A/B with small values (same for all states)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument('-v',"--verb", type=int, default=1)
    return parser.parse_args()

def group_lasso_penalty(fA, eps=1e-12):
    g = torch.sqrt(torch.sum(fA * fA, dim=0) + eps)
    diag_mask = ~torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
    return torch.sum(g[diag_mask])


def group_soft_threshold_(fA, lr, lam, eps=1e-12, epsW=0.0):
    g = torch.sqrt(torch.sum(fA * fA, dim=0) + eps)
    shrink = torch.clamp(1.0 - (lr * lam) / g, min=0.0)
    if epsW > 0.0:
        shrink = torch.where(g < epsW, torch.zeros_like(shrink), shrink)
    off_diag_mask = ~torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
    fA[:, off_diag_mask] *= shrink[off_diag_mask]


def spectral_project_(fA):
    rho_max = 0.0
    for m in range(fA.shape[0]):
        eigvals = torch.linalg.eigvals(fA[m])
        rho = torch.max(torch.abs(eigvals)).real.item()
        rho_max = max(rho_max, rho)
        if rho > 1.0:
            fA[m].div_(rho)
    return rho_max


def edge_stats(A, minW, include_offdiag=True):
    if torch.is_tensor(A):
        A_t = A
        if A_t.ndim == 2:
            A_t = A_t.unsqueeze(0)
        n_states, n_neurons = A_t.shape[0], A_t.shape[1]
        nz_total = int(torch.count_nonzero(A_t).item())
        nz_minw = int(torch.count_nonzero(torch.abs(A_t) > minW).item())
        stats = {
            "nz_total": nz_total,
            "nz_minw": nz_minw,
            "n_states": n_states,
            "n_neurons": n_neurons,
        }
        if include_offdiag:
            off_diag_mask = ~torch.eye(n_neurons, dtype=torch.bool, device=A_t.device)
            stats["nz_off"] = int(torch.count_nonzero(A_t[:, off_diag_mask]).item())
            stats["nz_minw_off"] = int(torch.count_nonzero(torch.abs(A_t[:, off_diag_mask]) > minW).item())
            stats["nz_off_per_state"] = stats["nz_off"] / float(n_states)
            stats["nz_minw_off_per_state"] = stats["nz_minw_off"] / float(n_states)
        return stats

    A_np = np.asarray(A)
    if A_np.ndim == 2:
        A_np = A_np[None, :, :]
    n_states, n_neurons = A_np.shape[0], A_np.shape[1]
    nz_total = int(np.count_nonzero(A_np))
    nz_minw = int(np.count_nonzero(np.abs(A_np) > minW))
    stats = {
        "nz_total": nz_total,
        "nz_minw": nz_minw,
        "n_states": n_states,
        "n_neurons": n_neurons,
    }
    if include_offdiag:
        off_diag_mask = ~np.eye(n_neurons, dtype=bool)
        stats["nz_off"] = int(np.count_nonzero(A_np[:, off_diag_mask]))
        stats["nz_minw_off"] = int(np.count_nonzero(np.abs(A_np[:, off_diag_mask]) > minW))
        stats["nz_off_per_state"] = stats["nz_off"] / float(n_states)
        stats["nz_minw_off_per_state"] = stats["nz_minw_off"] / float(n_states)
    return stats


def make_prism_out_base(fdr_name: str) -> str:
    base, aaa, rest = fdr_name.split("_", 2)
    hash6 = secrets.token_hex(3) # 6 hex digits
    return f"{base}_{aaa}-Estep-{hash6}"

def main():
    args = parse_args()
    args.inpPath = os.path.join(args.basePath, "lassoFdrFit")
    args.outPath = os.path.join(args.basePath, "prismFit")
    if args.minW < args.epsW:
        args.minW = args.epsW
    print('Args:',vars(args),'\n')
  
    os.makedirs(args.outPath, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = check_gpu_availability()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    fdrFF = os.path.join(args.inpPath, f"{args.fdrName}.FDRselected.npz")
    fdrD, fdrMD = read_data_npz(fdrFF, verb=args.verb > 0)
    assert isinstance(fdrMD, dict), "Expected metadata dict in FDRselected file"
    assert "provenance" in fdrMD, "Missing provenance in FDRselected metadata"

    A_avr = fdrD["A_avr"]
    B_avr = fdrD["B_avr"]
    E_mask = fdrD["E_mask"]

    assert A_avr.shape[0] == A_avr.shape[1], "A_avr must be square"
    assert B_avr.shape[0] == A_avr.shape[0], "B_avr size mismatch"
    assert E_mask.shape == A_avr.shape, "E_mask shape mismatch"

    mask_nonzero = (A_avr != 0)
    assert np.array_equal(mask_nonzero, E_mask), "A_avr nonzeros do not match E_mask"
    stats_inp = edge_stats(A_avr, args.minW, include_offdiag=True)
    input_edge_line = (
        f"M:input A_avr non-zero: {stats_inp['nz_total']} (includes {stats_inp['n_neurons']} diagonals), "
        f"off-diag: {stats_inp['nz_off']}, |A|>minW: {stats_inp['nz_minw']} "
        f"(off-diag: {stats_inp['nz_minw_off']})"
    )
    print(input_edge_line)
    
    prov = fdrMD["provenance"]
    assert "state_transition_file" in prov, "Missing state_transition_file in provenance"
    st_name = prov["state_transition_file"]

    spikesFF = os.path.join(args.basePath, "spikesData", f"{st_name}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 1)
    spikes = spikeD["spikes"]
    single_rates = spikeD["single_rates"]

    assert "time_step_sec" in spikeMD, "Missing time_step_sec in spikes metadata"
    assert "poisson_eta_clip" in spikeMD, "Missing poisson_eta_clip in spikes metadata"
    args.time_step_sec = float(spikeMD["time_step_sec"])
    args.eta_clip = float(spikeMD["poisson_eta_clip"])

    prismFF = os.path.join(args.basePath, "spikesData", f"{st_name}.prismTruth.npz")
    prismD, prismMD = read_data_npz(prismFF, verb=args.verb > 1)
    S_true = prismD["S_true"]
    assert "evol_conf" in prismMD, "Missing evol_conf in prismTruth metadata"
    assert "num_states" in prismMD["evol_conf"], "Missing num_states in evol_conf"
    args.num_states = int(prismMD["evol_conf"]["num_states"])


    T, N = spikes.shape
    assert S_true.shape[0] == T, "S_true length mismatch with spikes"

    # Build per-time data (t=1..T-1)
    Y_prev = torch.tensor(spikes[:-1], dtype=torch.float32, device=device)
    Y_curr = torch.tensor(spikes[1:], dtype=torch.float32, device=device)
    S_use = torch.tensor(S_true[1:], dtype=torch.long, device=device)
    T_eff = Y_prev.shape[0]

    # Initialize fA/fB for all states
    if args.rndStart:
        A0 = (0.01 * np.random.randn(N, N)).astype(np.float32)
        B0 = (0.01 * np.random.randn(N)).astype(np.float32)
        fA_init = np.repeat(A0[None, :, :], args.num_states, axis=0)
        fB_init = np.repeat(B0[None, :], args.num_states, axis=0)
    else:
        fA_init = np.repeat(A_avr[None, :, :], args.num_states, axis=0).astype(np.float32)
        fB_init = np.repeat(B_avr[None, :], args.num_states, axis=0).astype(np.float32)

    fA = torch.nn.Parameter(torch.tensor(fA_init, device=device))
    fB = torch.nn.Parameter(torch.tensor(fB_init, device=device))

    # Apply E_mask at init only
    E_mask_t = torch.tensor(E_mask, device=device, dtype=torch.bool)
    with torch.no_grad():
        fA.mul_(E_mask_t.unsqueeze(0))

    optimizer = torch.optim.Adam(
        [
            {"params": [fA], "lr": args.lr},
            {"params": [fB], "lr": args.lr},
        ]
    )
    log_dt = math.log(args.time_step_sec)
    lrA_start = float(args.lr)
    lrA_end = lrA_start * float(args.end_lr_frac)

    loss_epochs = []
    loss_dev = []
    loss_group = []
    loss_total = []
    rho_max_hist = []
    nz_edges_hist = []
    nz_edges_minW_hist = []
    nz_edges_minW_state_hist = []

    t_start = time.time()
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        if args.num_epochs > 1:
            lrA = lrA_start + (lrA_end - lrA_start) * (epoch - 1) / float(args.num_epochs - 1)
        else:
            lrA = lrA_start
        optimizer.param_groups[0]["lr"] = lrA
        optimizer.param_groups[1]["lr"] = lrA
        optimizer.zero_grad(set_to_none=True)
        dev_sum = 0.0

        for t0_idx in range(0, T_eff, args.chunk_size):
            t1_idx = min(T_eff, t0_idx + args.chunk_size)
            y_prev = Y_prev[t0_idx:t1_idx]
            y_curr = Y_curr[t0_idx:t1_idx]
            s_idx = S_use[t0_idx:t1_idx]

            A_t = fA[s_idx]
            B_t = fB[s_idx]
            eta = torch.einsum("kij,kj->ki", A_t, y_prev) + B_t
            eta_clip = torch.clamp(eta, max=args.eta_clip) 
            lam = torch.exp(eta_clip) * args.time_step_sec
            dev = lam - y_curr * (eta + log_dt)
            loss_chunk = dev.sum() / float(y_curr.shape[0])
            loss_chunk.backward()
            dev_sum += loss_chunk.detach().item() * float(y_curr.shape[0])

        optimizer.step()

        with torch.no_grad():
            group_soft_threshold_(fA, lrA, args.lambda3, epsW=args.epsW)
            rho_max = spectral_project_(fA)
            group_val = group_lasso_penalty(fA).item()
            stats_epoch = edge_stats(fA, args.minW, include_offdiag=True)
            nz_edges = stats_epoch["nz_total"]
            nz_edges_minW = stats_epoch["nz_minw"]
            off_diag_mask = ~torch.eye(fA.shape[1], dtype=torch.bool, device=fA.device)
            edges_minW_state = torch.count_nonzero((torch.abs(fA) > args.minW) & off_diag_mask, dim=(1, 2))

        dev_mean = dev_sum / float(T_eff)
        total_loss = dev_mean + args.lambda3 * group_val
        elapsed = time.time() - t0
        elaT = time.time() - t_start
        if epoch % 10 == 1:
            edges_per_state = stats_epoch["nz_off_per_state"]
            edges_per_state_minW = stats_epoch["nz_minw_off_per_state"]
            print(
                f"epoch {epoch:3d}  dev={dev_mean:.4e}  grp={group_val:.4e}  tot={total_loss:.4e}  "
                f"edges/state={edges_per_state:.1f}, {edges_per_state_minW:.1f}>|minW|  elaT={elaT:.1f}s"
            )

        loss_epochs.append(epoch)
        loss_dev.append(dev_mean)
        loss_group.append(group_val)
        loss_total.append(total_loss)
        rho_max_hist.append(rho_max)
        nz_edges_hist.append(nz_edges)
        nz_edges_minW_hist.append(nz_edges_minW)
        nz_edges_minW_state_hist.append(edges_minW_state.detach().cpu().numpy().astype(np.int64))


    # Save results
    out_base = make_prism_out_base(args.fdrName)
    out_name = f"{out_base}.prismMstep.npz"
    outFF = os.path.join(args.outPath, out_name)

    outD = {
        "fA": fA.detach().cpu().numpy(),
        "fB": fB.detach().cpu().numpy(),
        "E_mask": E_mask,
        "single_rates": single_rates,
        "loss_epochs": np.asarray(loss_epochs, dtype=np.int32),
        "loss_dev": np.asarray(loss_dev, dtype=np.float64),
        "loss_group": np.asarray(loss_group, dtype=np.float64),
        "loss_total": np.asarray(loss_total, dtype=np.float64),
        "rho_max": np.asarray(rho_max_hist, dtype=np.float64),
        "nz_edges": np.asarray(nz_edges_hist, dtype=np.int64),
        "nz_edges_minW": np.asarray(nz_edges_minW_hist, dtype=np.int64),
        "nz_edges_minW_state": np.stack(nz_edges_minW_state_hist, axis=0),
    }

    stats_final = edge_stats(fA, args.minW, include_offdiag=True)
    edges_per_state = stats_final["nz_total"] / float(stats_final["n_states"])
    edges_per_state_minw = stats_final["nz_minw"] / float(stats_final["n_states"])
    print(input_edge_line)
    print(
        "M:final A_hat non-zero: %d (includes %d diagonals), off-diag: %.1f, "
        "|A|>minW: %d (off-diag: %.1f), edges/state: %.1f, edges/state>|minW|=%.1f"
        % (
            stats_final["nz_total"],
            stats_final["n_neurons"],
            stats_final["nz_off_per_state"],
            stats_final["nz_minw"],
            stats_final["nz_minw_off_per_state"],
            edges_per_state,
            edges_per_state_minw,
        )
    )

    outMD = dict(fdrMD)
    outMD["fit_type"] = "prismMstep"
    outMD["train"] = {
        "num_epochs": int(args.num_epochs),
        "lr": float(args.lr),
        "end_lr_frac": float(args.end_lr_frac),
        "lambda3": float(args.lambda3),
        "minW": float(args.minW),
        "chunk_size": int(args.chunk_size),
        "seed": int(args.seed),
        "time_step_sec": float(args.time_step_sec),
        "eta_clip": float(args.eta_clip),
        "num_states": int(args.num_states),
        "num_neurons": int(N),
        "num_steps": int(T),
    }
    outMD["provenance"] = prov
    outMD["evol_conf"] = prismMD["evol_conf"]
    outMD["dale_conf"] = prismMD.get("dale_conf", {})

    write_data_npz(outD, outFF, metaD=outMD)
    print(f"Saved prism M-step fit to: {outFF}")
    print('   basePath='+args.basePath)
    print('   ./prism_Mstep_eval.py  --basePath $basePath  --dataName %s   -p a b f  \n ' % (out_base))

if __name__ == "__main__":
    main()
