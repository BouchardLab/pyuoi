#!/usr/bin/env python3
"""
PRISM Stage 3 - Dictionary Update (train).

Goal: Given fixed C_hat from Stage 2, optimize A (M,N,N) and B (M,N)
to minimize the Poisson NLL subject to:
  - Dale's law  : row i of A_m keeps its sign for all m
                  (excitatory i < num_excite, inhibitory i >= num_excite)
  - Spectral radius soft penalty: penalize rho(A_m) > rho_max
                  via power iteration (differentiable, stable)
  - Optional L1 sparsity penalty on A entries
  - Optional E_true mask (upper bound benchmark only)

Reads:
  <basePath>/prismFit/<dataName>.stage2.npz
  <basePath>/spikesData/<spikesName>.spikes.npz
  <basePath>/truthDale/<truthName>.simTruth.npz
Writes:
  <basePath>/prismFit/<dataName>_<hash6>.stage3.npz
"""

import os
import argparse
import time
import hashlib
from pprint import pprint
import numpy as np
import torch

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from Util_stage3 import (
    make_dale_mask, project_dale,
    spectral_radius_approx, make_hard_C,
    print_dale_diagnostics,
    linear_lr, set_lr,
    LOG_EPS, ETA_CLIP,
)

# ================================================================
#  Argument parser
# ================================================================
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        dest="verb")
    parser.add_argument("--basePath",
                        default="/dataVault2026/neurodata_tmp2")
    parser.add_argument("--dataName", default=None,
                        help="Stage 2 output name")
    parser.add_argument("--device", default="cuda")
    # optimiser
    parser.add_argument("--lr",      type=float, default=1e-3,
                        help="Learning rate for NLL term (sets both if "
                             "--lr_nll / --lr_pen not given).")
    parser.add_argument("--lr_nll",  type=float, default=None,
                        help="Learning rate for NLL term only. "
                             "Overrides --lr for NLL.")
    parser.add_argument("--lr_pen",  type=float, default=None,
                        help="Learning rate for spectral-radius penalty term only. "
                             "Overrides --lr for penalty.")
    parser.add_argument("--end_lr_frac", type=float, default=0.2,
                        help="Final LR as fraction of initial LR "
                             "(linear decay). Default 0.2 = 1/5.")
    parser.add_argument("--n_epoch", type=int,   default=120)
    parser.add_argument("--batch",   type=int,   default=500,
                        help="Time steps per mini-batch.")
    # constraints
    parser.add_argument("--lam_rho", type=float, default=0.5,
                        help="Spectral radius penalty weight.")
    parser.add_argument("--lam_l1",  type=float, default=1e-6,
                        help="L1 sparsity penalty weight on A entries.")
    parser.add_argument("--time_steps_range", type=int, nargs=2,
                        default=None)
    parser.add_argument("--hard_C", action="store_true", default=False)
    parser.add_argument("--stable_thresh", type=float, default=0.65)
    parser.add_argument("--rho_max", type=float, default=0.97)
    parser.add_argument("--use_e_mask", action="store_true", default=False)
    parser.add_argument("--init_rho", type=float, default=0.40)

    args = parser.parse_args()

    # resolve lr_nll / lr_pen — fall back to --lr if not set
    if args.lr_nll is None:
        args.lr_nll = args.lr
    if args.lr_pen is None:
        args.lr_pen = args.lr

    args.inpTruth  = os.path.join(args.basePath, "truthDale")
    args.inpSpikes = os.path.join(args.basePath, "spikesData")
    args.inpFit    = os.path.join(args.basePath, "prismFit")
    args.outPath   = os.path.join(args.basePath, "prismFit")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert args.dataName is not None, "must provide --dataName"
    assert os.path.exists(args.basePath), \
        f"missing basePath: {args.basePath}"
    os.makedirs(args.outPath, exist_ok=True)
    return args


# ================================================================
#  Per-state training loop
# ================================================================
def run_dict_update_perstate(
        A_init, B_init, state_hat, C_hat,
        spikes, dt, eta_clip, dale_sign,
        lam_rho, rho_max, stable_thresh,
        lr_nll, lr_pen, end_lr_frac,
        n_epoch, batch_size, device,
        e_mask=None, lam_l1=0.0, verb=1):
    """
    Optimize each A_m, B_m using only stable time steps where
    argmax(C_hat[t]) == m  AND  max(C_hat[t]) > stable_thresh.

    Two independent optimizers per mode:
      opt_nll : updates [Am, Bm]  with lr_nll  (NLL term)
      opt_pen : updates [Am]      with lr_pen   (spectral penalty + L1)
    Both use persistent state across epochs (no re-creation per epoch).
    LR decays linearly from initial value to initial * end_lr_frac.
    """
    T, N = spikes.shape
    M    = A_init.shape[0]

    if not isinstance(A_init, torch.Tensor):
        A_init = torch.tensor(A_init, dtype=torch.float32)
    if not isinstance(B_init, torch.Tensor):
        B_init = torch.tensor(B_init, dtype=torch.float32)

    Y_dev     = spikes.float().to(device)
    state_dev = torch.tensor(state_hat, dtype=torch.long, device=device)
    c_max_dev = torch.tensor(
        C_hat.max(dim=1).values.numpy(), device=device)

    # stable time-step indices per state (computed once)
    t_idx_per_state = []
    for m in range(M):
        mask = (state_dev == m) & (c_max_dev > stable_thresh)
        t_idx_per_state.append(torch.where(mask)[0])
        if verb > 0:
            print(f"  state {m}: {len(t_idx_per_state[m])} stable bins "
                  f"({100*len(t_idx_per_state[m])/T:.1f}% of T)")

    # ── persistent parameters and optimizers (created once) ──────────
    Am_list      = []
    Bm_list      = []
    opt_nll_list = []
    opt_pen_list = []

    for m in range(M):
        Am = A_init[m].clone().to(device).requires_grad_(True)
        Bm = B_init[m].clone().to(device).requires_grad_(True)
        Am_list.append(Am)
        Bm_list.append(Bm)
        opt_nll_list.append(torch.optim.Adam([Am, Bm], lr=lr_nll))
        opt_pen_list.append(torch.optim.Adam([Am],      lr=lr_pen))

    loss_hist = torch.zeros(n_epoch)
    nll_hist  = torch.zeros(n_epoch)
    pen_hist  = torch.zeros(n_epoch)
    l1_hist   = torch.zeros(n_epoch)

    t0_wall = time.time()

    for epoch in range(n_epoch):

        # ── linear LR decay ──────────────────────────────────────────
        cur_lr_nll = linear_lr(lr_nll, end_lr_frac, epoch, n_epoch)
        cur_lr_pen = linear_lr(lr_pen, end_lr_frac, epoch, n_epoch)
        for m in range(M):
            set_lr(opt_nll_list[m], cur_lr_nll)
            set_lr(opt_pen_list[m], cur_lr_pen)

        ep_nll   = 0.0
        ep_pen   = 0.0
        ep_l1    = 0.0
        mode_nll = []
        mode_pen = []
        mode_l1  = []

        for m in range(M):
            t_idx = t_idx_per_state[m]
            T_m   = len(t_idx)
            if T_m == 0:
                print(f"  WARNING: no stable bins for state {m}, skipping")
                mode_nll.append(0.0)
                mode_pen.append(0.0)
                mode_l1.append(0.0)
                continue

            Am      = Am_list[m]
            Bm      = Bm_list[m]
            opt_nll = opt_nll_list[m]
            opt_pen = opt_pen_list[m]
            dale_m  = dale_sign[m]

            perm    = torch.randperm(T_m, device=device)
            n_batch = max(1, T_m // batch_size)

            m_nll = 0.0
            m_pen = 0.0
            m_l1  = 0.0

            for bi in range(n_batch):
                idx_b = perm[bi * batch_size:
                             min(T_m, (bi + 1) * batch_size)]
                t_b   = t_idx[idx_b]
                n_b   = len(t_b)

                # ── NLL + L1 backward (updates Am and Bm) ─────────────────
                opt_nll.zero_grad()
                total = torch.tensor(0.0, device=device)
                for t in t_b:
                    t_int  = int(t.item())
                    Y_prev = Y_dev[t_int - 1] if t_int > 0 \
                             else torch.zeros(N, device=device)
                    Y_t    = Y_dev[t_int]
                    eta    = torch.clamp(Am @ Y_prev + Bm,
                                         min=-eta_clip, max=eta_clip)
                    lam    = torch.exp(eta) * dt
                    total = total + (lam - Y_t * torch.log(lam + LOG_EPS)).sum()
                nll_m = total / n_b
                l1_m  = lam_l1 * torch.sum(torch.abs(Am))
                (nll_m + l1_m).backward()
                torch.nn.utils.clip_grad_norm_([Am, Bm], max_norm=5.0)
                opt_nll.step()

                # ── Penalty backward (updates Am only) ──────────────
                opt_pen.zero_grad()
                ev_max = spectral_radius_approx(Am)
                pen_m  = lam_rho * torch.relu(ev_max - rho_max) ** 2
                pen_m.backward()
                torch.nn.utils.clip_grad_norm_([Am], max_norm=5.0)
                opt_pen.step()

                # ── Dale projection + optional E_mask ────────────────
                project_dale(Am, dale_m)
                if e_mask is not None:
                    with torch.no_grad():
                        Am.data *= e_mask

                m_nll += nll_m.item()
                m_pen += pen_m.item()
                m_l1  += l1_m.item()

            # write back (Am still lives on device as persistent tensor)
            ep_nll += m_nll / n_batch
            ep_pen += m_pen / n_batch
            ep_l1  += m_l1  / n_batch
            mode_nll.append(m_nll / n_batch)
            mode_pen.append(m_pen / n_batch)
            mode_l1.append(m_l1  / n_batch)

        ep_nll /= M
        ep_pen /= M
        ep_l1  /= M
        loss_hist[epoch] = ep_nll + ep_pen + ep_l1
        nll_hist[epoch]  = ep_nll
        pen_hist[epoch]  = ep_pen
        l1_hist[epoch]   = ep_l1

        if verb > 0 and epoch%2==1 :
            with torch.no_grad():
                rhos = [torch.linalg.eigvals(Am_list[m]).abs().max().item()
                        for m in range(M)]
            rho_str      = "  ".join(
                [f"ρ{m}={r:.3f}" for m, r in enumerate(rhos)])
            per_mode_str = "  ".join(
                [f"[m{m}: nll={mode_nll[m]:.2f} pen={mode_pen[m]:.2e}"
                 f" l1={mode_l1[m]:.2e}]" for m in range(M)])
            print(f"  epoch {epoch+1:3d}/{n_epoch}  "
                  f"nll={ep_nll:.4f}  pen={ep_pen:.2e}  l1={ep_l1:.5f}  "
                  f"lr_nll={cur_lr_nll:.2e}  lr_pen={cur_lr_pen:.2e}  "
                  f"{rho_str}  elaT={time.time()-t0_wall:.1f}s")
            print(f"    {per_mode_str}")

    # collect results from persistent tensors
    A_result = torch.stack([Am_list[m].detach().cpu() for m in range(M)])
    B_result = torch.stack([Bm_list[m].detach().cpu() for m in range(M)])

    return {
        "A_hat":      A_result,
        "B_hat":      B_result,
        "loss_epoch": loss_hist,
        "nll_epoch":  nll_hist,
        "pen_epoch":  pen_hist,
        "l1_epoch":   l1_hist,
    }


# ================================================================
#  Main
# ================================================================
def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    # ── load stage 2 ─────────────────────────────────────────────────
    s2FF = os.path.join(args.inpFit, f"{args.dataName}.stage2.npz")
    s2D, s2MD = read_data_npz(s2FF, verb=args.verb > 0)

    spikes_name = s2MD["input_spikes_name"]
    truth_name  = s2MD["input_truth_name"]
    dt          = float(s2MD["time_step_sec"])
    eta_clip    = float(s2MD["eta_clip"])
    state_hat   = s2D["state_hat"].astype(np.int32)

    spikesFF = os.path.join(args.inpSpikes, f"{spikes_name}.spikes.npz")
    spikesD, _ = read_data_npz(spikesFF, verb=args.verb > 0)

    truthFF = os.path.join(args.inpTruth, f"{truth_name}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)

    A_true = torch.tensor(trueD["A_true"], dtype=torch.float32)
    B_true = torch.tensor(trueD["B_true"], dtype=torch.float32)
    C_hat  = torch.tensor(s2D["C_hat"],    dtype=torch.float32)
    spikes = torch.tensor(spikesD["spikes"], dtype=torch.int32)

    if A_true.ndim == 2: A_true = A_true.unsqueeze(0)
    if B_true.ndim == 1: B_true = B_true.unsqueeze(0)

    M, N, _ = A_true.shape
    T       = spikes.shape[0]
    num_excite = int(trueMD["dale_conf"]["num_excite"])

    C_hat_train = make_hard_C(C_hat) if args.hard_C else C_hat

    if args.verb > 0:
        print(f"\nA={tuple(A_true.shape)}  B={tuple(B_true.shape)}  "
              f"C_hat={tuple(C_hat.shape)}  spikes={tuple(spikes.shape)}")
        print(f"num_excite={num_excite}  dt={dt}  eta_clip={eta_clip}")

    # ── optional time slice ───────────────────────────────────────────
    if args.time_steps_range is not None:
        t_lo, t_hi = args.time_steps_range
        t_lo = max(0, t_lo); t_hi = min(T, t_hi)
        spikes      = spikes[t_lo:t_hi]
        C_hat_train = C_hat_train[t_lo:t_hi]
        state_hat   = state_hat[t_lo:t_hi]
        T = spikes.shape[0]
        print(f"Sliced to time bins [{t_lo}, {t_hi}), T={T}")

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    if args.verb > 0:
        print(f"Using device: {device}")

    dale_sign = make_dale_mask(num_excite, N, M, device)

    # ── Dale check on A_true ─────────────────────────────────────────
    with torch.no_grad():
        violation = (dale_sign * A_true.to(device) < 0).float().sum().item()
        print(f"Dale violations in A_true: {int(violation)}")
    print_dale_diagnostics(trueD["E_true"], A_true, num_excite,
                           label="A_true")

    # ── E_mask ───────────────────────────────────────────────────────
    if args.use_e_mask:
        E_mask = torch.tensor(
            (trueD["E_true"] != 0).astype(np.float32),
            dtype=torch.float32, device=device)
        if args.verb > 0:
            n_edges = int(E_mask.sum().item())
            print(f"Using E_true mask: {n_edges}/{N*N} edges "
                  f"({100*n_edges/(N*N):.1f}%)")
    else:
        E_mask = None

    # ── A initialisation ─────────────────────────────────────────────
    single_rates = spikesD["single_rates"]
    B_init = np.tile(
        np.log(np.maximum(single_rates, 1e-6)), (M, 1)
    ).astype(np.float32)

    if args.init_rho is not None:
        A_init = np.random.randn(*A_true.shape).astype(np.float32) * 0.01
        for m in range(A_init.shape[0]):
            Am  = torch.tensor(A_init[m])
            rho = torch.linalg.eigvals(Am).abs().max().item()
            if rho > 1e-6:
                A_init[m] *= args.init_rho / rho
    else:
        A_init = A_true.numpy()

    # ── training ─────────────────────────────────────────────────────
    t0 = time.time()
    results = run_dict_update_perstate(
        A_init        = A_init,
        B_init        = B_init,
        state_hat     = state_hat,
        C_hat         = C_hat_train,
        spikes        = spikes,
        dt            = dt,
        eta_clip      = eta_clip,
        dale_sign     = dale_sign,
        lam_rho       = args.lam_rho,
        rho_max       = float(args.rho_max),
        stable_thresh = args.stable_thresh,
        lr_nll        = args.lr_nll,
        lr_pen        = args.lr_pen,
        end_lr_frac   = args.end_lr_frac,
        n_epoch       = args.n_epoch,
        batch_size    = args.batch,
        device        = device,
        e_mask        = E_mask,
        lam_l1        = args.lam_l1,
        verb          = args.verb,
    )
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} sec")

    print_dale_diagnostics(trueD["E_true"], results["A_hat"],
                           num_excite, label="A_hat")

    # ── save ─────────────────────────────────────────────────────────
    hash6   = hashlib.md5(os.urandom(32)).hexdigest()[:6]
    outName = f"{args.dataName}_{hash6}"

    outD = {
        "A_hat":      results["A_hat"].numpy().astype(np.float32),
        "B_hat":      results["B_hat"].numpy().astype(np.float32),
        "A_true":     A_true.numpy().astype(np.float32),
        "B_true":     B_true.numpy().astype(np.float32),
        "loss_epoch": results["loss_epoch"].numpy().astype(np.float32),
        "nll_epoch":  results["nll_epoch"].numpy().astype(np.float32),
        "pen_epoch":  results["pen_epoch"].numpy().astype(np.float32),
        "l1_epoch":   results["l1_epoch"].numpy().astype(np.float32),
    }
    outMD = {
        "data_type":         "prismStage3",
        "short_name":        outName,
        "input_stage2_name": args.dataName,
        "input_spikes_name": spikes_name,
        "input_truth_name":  truth_name,
        "time_step_sec":     dt,
        "stage":             3,
        "eta_clip":          eta_clip,
        "num_neurons":       int(N),
        "num_steps":         int(T),
        "num_states":        int(M),
        "num_excite":        num_excite,
        "device":            str(device),
        "lr_nll":            args.lr_nll,
        "lr_pen":            args.lr_pen,
        "end_lr_frac":       args.end_lr_frac,
        "n_epoch":           args.n_epoch,
        "batch_size":        args.batch,
        "lam_rho":           args.lam_rho,
        "lam_l1":            args.lam_l1,
        "rho_max":           args.rho_max,
        "train_time_sec":    round(elapsed, 1),
        "time_steps_range":  args.time_steps_range,
        "final_nll":         float(results["nll_epoch"][-1].item()),
        "final_pen":         float(results["pen_epoch"][-1].item()),
        "final_l1":          float(results["l1_epoch"][-1].item()),
        "hard_C":            args.hard_C,
        "use_e_mask":        args.use_e_mask,
        "stable_thresh":     args.stable_thresh,
    }

    outFF = os.path.join(args.outPath, f"{outName}.stage3.npz")
    write_data_npz(outD, outFF, metaD=outMD)

    if args.verb > 1:
        print("\nstage3 metadata:"); pprint(outMD)

    print(f"\n  ./prism_stage3_eval.py  --basePath $basePath"
          f"  --dataName {outName}")


if __name__ == "__main__":
    main()
