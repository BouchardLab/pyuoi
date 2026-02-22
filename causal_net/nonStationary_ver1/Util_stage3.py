"""
Utility functions for PRISM Stage 3:
  - Dale's law mask / projection
  - Spectral radius (power iteration)
  - Diagnostics printing
  - Hard-C assignment
"""

import numpy as np
import torch

LOG_EPS  = 1e-10
ETA_CLIP = 20.0


# ================================================================
#  Dale's law
# ================================================================
def make_dale_mask(num_excite, N, M, device):
    """
    Build sign mask dale_sign (M, N, N):
      +1 where A should be >= 0  (rows 0..num_excite-1)
      -1 where A should be <= 0  (rows num_excite..N-1)
    A[m, i, j] = weight from neuron j to neuron i,
    Dale's law constrains rows (post-synaptic? No — source = row i).
    """
    sign = torch.ones(N, dtype=torch.float32)
    sign[num_excite:] = -1.0
    dale_sign = sign.unsqueeze(0).unsqueeze(-1).expand(M, N, N).to(device)
    return dale_sign


def project_dale(Am, dale_m):
    """
    Hard-project Am onto Dale's law in-place.
    Zeros any entry with the wrong sign.
    """
    with torch.no_grad():
        Am.data = torch.relu(dale_m * Am) * dale_m


# ================================================================
#  Spectral radius (differentiable, power iteration)
# ================================================================
def spectral_radius_approx(A, n_iter=20):
    """
    Approximate spectral radius via power iteration.
    Fully differentiable, real-valued, stable gradients.
    """
    v = torch.randn(A.shape[0], 1, device=A.device)
    v = v / v.norm()
    for _ in range(n_iter):
        v = A @ v
        norm = v.norm()
        v = v / norm
    return (A @ v).norm() / v.norm()


# ================================================================
#  Hard assignment from C_hat
# ================================================================
def make_hard_C(C_hat):
    hard = torch.zeros_like(C_hat)
    idx  = C_hat.argmax(dim=1)
    hard.scatter_(1, idx.unsqueeze(1), 1.0)
    return hard


# ================================================================
#  Dale diagnostics
# ================================================================
def print_dale_diagnostics(E_true, A, num_excite, label="A"):
    """
    E_true:     (N, N) binary presence mask
    A:          (M, N, N) matrix to evaluate — numpy or torch
    num_excite: first num_excite rows are excitatory sources
    """
    if isinstance(A, torch.Tensor):
        A = A.numpy()

    rows, cols = np.where(E_true != 0)
    offdiag    = [(i, j) for i, j in zip(rows, cols) if i != j]
    diag_count = int(np.sum(E_true != 0)) - len(offdiag)

    print(f"Dale diagnostics — {label}")
    print(f"  {'mode':<8} {'exc_TP':>8} {'exc_FP':>8} {'exc_FN':>8}"
          f" {'inh_TP':>8} {'inh_FP':>8} {'inh_FN':>8}")
    print("  " + "-" * 66)

    for m in range(A.shape[0]):
        exc_TP = exc_FP = exc_FN = 0
        inh_TP = inh_FP = inh_FN = 0

        for (i, j) in offdiag:
            is_exc    = i < num_excite
            val       = float(A[m, i, j])
            pred_sign = 1 if val > 0 else (-1 if val < 0 else 0)

            if is_exc:
                if pred_sign > 0:   exc_TP += 1
                elif pred_sign < 0: exc_FP += 1
                else:               exc_FN += 1
            else:
                if pred_sign < 0:   inh_TP += 1
                elif pred_sign > 0: inh_FP += 1
                else:               inh_FN += 1

        print(f"  {m:<8} {exc_TP:>8} {exc_FP:>8} {exc_FN:>8}"
              f" {inh_TP:>8} {inh_FP:>8} {inh_FN:>8}")

    if diag_count > 0:
        print(f"(Excluded {diag_count} diagonal self-loop(s) from reporting)")


# ================================================================
#  LR schedule
# ================================================================
def linear_lr(lr_init, end_lr_frac, epoch, n_epoch):
    """
    Linear decay from lr_init to lr_init * end_lr_frac over n_epoch steps.
    epoch is 0-indexed.
    """
    if n_epoch <= 1:
        return lr_init
    frac = 1.0 - (1.0 - end_lr_frac) * epoch / (n_epoch - 1)
    return lr_init * frac


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr
