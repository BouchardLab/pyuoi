#!/usr/bin/env python3
"""
Topology analysis for simulated Dale Poisson network data.

Loads simTruth.npz from gen_daleMatrices4.py.  From the weighted off-diagonal
matrix A_off_true, builds the binary adjacency G (doc/network_ver4_topoDiscovery.tex,
Eq. adjacency): G_ij = 1 iff i != j and A_off_ij != 0 (directed edge
presynaptic j -> postsynaptic i).

Method 1 — degree assortativity (Newman, directed): Pearson r between
k_i^in at the target and k_j^out at the source over all edges (Eq. assortativity).

Usage:
    ./topoAna_daleMatrix4.py --basePath $basePath --dataName daleN200_fa3733
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
from pprint import pprint
import numpy as np
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
import argparse


def adjacency_from_A_off(A_off, atol=0.0):
    """
    Binary directed adjacency G: G_ij = 1 if |A_off_ij| > atol and i != j.
    Rows index postsynaptic i, columns presynaptic j (matches gen_daleMatrices4 / TeX).
    """
    A = np.asarray(A_off, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A_off must be square (N, N)")
    G = (np.abs(A) > atol).astype(np.float64)
    np.fill_diagonal(G, 0.0)
    return G


def shuffle_neuron_order(A_off, perm=None):
    """
    Reindex neurons with a shared permutation on rows and columns.
    Rows are shuffled first; columns are reordered to the same new neuron order.
    Returns (A_off_shuffled, perm).
    """
    A = np.asarray(A_off, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A_off must be square (N, N)")

    n = A.shape[0]
    if perm is None:
        perm = np.random.permutation(n)
    else:
        perm = np.asarray(perm)
        if perm.shape != (n,):
            raise ValueError("perm must have shape (N,)")
    return A[perm][:, perm], perm


def directed_degree_assortativity(G):
    """
    Directed degree assortativity r per network_ver4_topoDiscovery.tex, Sec. 1.
    For each edge j -> i (G_ij = 1): pair (k_i^in, k_j^out).
    Uses |E|^{-1} moments (population cov / product of population std devs).
    Returns (r, |E|, extra dict).  r is nan if |E| < 2 or a std dev is 0.
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    k_in = G.sum(axis=1)
    k_out = G.sum(axis=0)
    ei, ej = np.nonzero(G > 0)
    m = int(ei.size)
    if m < 2:
        return float("nan"), m, {"n_nodes": n, "n_edges": m, "reason": "|E| < 2"}

    x = k_in[ei].astype(np.float64) # Target in-degrees
    y = k_in[ej].astype(np.float64) # Source in-degrees (In-In correlation)
    mx = x.mean()
    my = y.mean()
    cov = (x * y).mean() - mx * my
    vx = ((x - mx) ** 2).mean()
    vy = ((y - my) ** 2).mean()
    sig_in_target = float(np.sqrt(vx))
    sig_in_source = float(np.sqrt(vy))
    if sig_in_target <= 0.0 or sig_in_source <= 0.0:
        return float("nan"), m, {
            "n_nodes": n,
            "n_edges": m,
            "sigma_in_target": sig_in_target,
            "sigma_in_source": sig_in_source,
            "reason": "zero std on edge multiset",
        }
    r = cov / (sig_in_target * sig_in_source)
    return float(r), m, {
        "n_nodes": n,
        "n_edges": m,
        "r_assortativity": float(r),
        "mean_k_in_target": float(mx),
        "mean_k_in_source": float(my),
        "sigma_in_target": sig_in_target,
        "sigma_in_source": sig_in_source,
        "cov_edge": float(cov),
    }


def run_method1(G, dmd, verb=1):
    r, m_edges, info = directed_degree_assortativity(G)
    d_ker = dmd["placement_ker_delta"]

    print("\n--- Method 1: directed degree assortativity  ---")
    print("  N=%d  |E|=%d  r=%s" % (info["n_nodes"], m_edges, repr(r) if np.isnan(r) else "%.6f" % r))
    if verb > 1:
        print("  mean target k_in on edges:  %.6f" % info["mean_k_in_target"])
        print("  mean source k_in on edges:  %.6f" % info["mean_k_in_source"])
        print("  sigma target in:  %.6f" % info["sigma_in_target"])
        print("  sigma source in:  %.6f" % info["sigma_in_source"])
        print("  (truth metadata) placement_ker_delta = %s" % (d_ker,))

    return info


def run_method2(G, dmd, verb=1):
    G = np.asarray(G, dtype=np.float64)
    G2 = G @ G
    k_in = G.sum(axis=1) # Target nodes (rows)
    k_out = G.sum(axis=0) # Source nodes (columns)
    
    ei, ej = np.nonzero(G > 0)
    m = int(ei.size)
    if m == 0:
        return {"mean_jaccard": float("nan"), "n_edges": 0, "reason": "no edges"}

    num = G2[ei, ej]
    den = k_in[ei] + k_out[ej] - num
    # den is always >= 2 because i!=j, j in N_in(i), i in N_out(j)
    j_idx = num / den
    j_mean = float(np.mean(j_idx))
    
    print("\n--- Method 2: common-neighbor overlap (Jaccard Index) ---")
    print("  mean Jaccard index: %.6f" % j_mean)
    
    info = {
        "mean_jaccard": j_mean,
        "n_edges": m
    }
    return info


def run_method3(G, dmd, verb=1):
    G = np.asarray(G, dtype=np.float64)
    K = 6
    k_vals = np.arange(1, K + 1) # k = 1..K
    
    # Tk = trace(G^(k+1)) / sum(G^k)
    transitivity = []
    Gk = G.copy()
    for k in k_vals:
        Gnext = Gk @ G
        num = float(np.trace(Gnext))
        den = float(Gk.sum())
        if den > 0:
            transitivity.append(num / den)
        else:
            transitivity.append(0.0)
        Gk = Gnext
    
    transitivity = np.array(transitivity)
    
    # xi = -d log Tk / dk for k in 2..K
    # Handle log compatibility
    fit_mask = (k_vals >= 2) & (transitivity > 1e-16)
    xi = float("nan")
    if np.sum(fit_mask) >= 2:
        slope, _ = np.polyfit(k_vals[fit_mask], np.log(transitivity[fit_mask]), 1)
        xi = -float(slope)
        
    print("\n--- Method 4: cycle closure decay rate ---")
    print("  decay rate xi: %.6f" % xi)
    
    return {
        "cycle_decay": xi,
        "transitivity_k": transitivity.tolist()
    }


def run_method4(A_off, dmd, verb=1):
    try:
        import gudhi
    except ImportError as exc:
        raise ImportError(
            "Method 4 requires GUDHI. Install it in this environment before running topology analysis."
        ) from exc
    A = np.abs(A_off)
    # Symmetrized interaction strength (LaTeX Method 5, weighted variant)
    W = (A + A.T) / 2.0
    N = W.shape[0]
    
    st = gudhi.SimplexTree()
    # Add nodes at filtration 0
    for i in range(N):
        st.insert([i], filtration=0.0)
    
    # Add edges with filtration -W_ij (strongest edges enter first)
    ei, ej = np.nonzero(W > 1e-12)
    for i, j in zip(ei, ej):
        if i < j:
            st.insert([i, j], filtration=-float(W[i, j]))
            
    # Expand to Rips complex of dimension 2 to capture H1 (loops)
    st.expansion(2)
    
    # Compute persistence
    st.persistence()

    # Total persistence Pi_1 (dim 1)
    diag1 = st.persistence_intervals_in_dimension(1)
    pi1 = 0.0
    if len(diag1) > 0:
        # Filter out infinite death
        valid1 = diag1[np.isfinite(diag1[:, 1])]
        pi1 = float(np.sum(valid1[:, 1] - valid1[:, 0]))
    
    print("\n--- Method 4: persistent homology (H1) ---")
    print("  total persistence Pi_1: %.6f" % pi1)
    
    return {
        "homology_k1": pi1,
        "n_features_k1": len(diag1)
    }


def get_parser():
    parser = argparse.ArgumentParser(description="Topology analysis on Dale simTruth.npz (method 1: assortativity)")
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=1, dest="verb")
    parser.add_argument("--basePath", default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input data")
    parser.add_argument("--dataName", default="daleN150_448b86", help="simulated Dale network base name")
    parser.add_argument("--shuffle", action="store_true", help="randomize neuron order of A_off before computing G")

    args = parser.parse_args()

    args.inpPath = os.path.join(args.basePath, "truthDale")
    args.outPath = os.path.join(args.basePath, "topoAna")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert os.path.exists(args.basePath)
    os.makedirs(args.outPath, exist_ok=True)
    return args


if __name__ == "__main__":
    args = get_parser()
    np.set_printoptions(precision=3)

    truthFF = os.path.join(args.inpPath, f"{args.dataName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nSimulation Truth Metadata:")
        pprint(trueMD)

    trueMD["short_name"] = args.dataName

    A_off = np.asarray(trueD["A_off_true"], dtype=np.float64)
    print("read obj: A_off_true %s %s" % (A_off.shape, A_off.dtype))
    neuron_perm = np.arange(A_off.shape[0], dtype=np.int64)
    if args.shuffle:
        A_off, neuron_perm = shuffle_neuron_order(A_off)
        print("applied neuron shuffle to A_off, first 10 perm entries:", neuron_perm[:10])

    G = adjacency_from_A_off(A_off)
    dmd = trueMD["dale_conf"]
    
    info1 = run_method1(G, dmd, verb=args.verb)
    info2 = run_method2(G, dmd, verb=args.verb)
    info3 = run_method3(G, dmd, verb=args.verb)
    info4 = run_method4(A_off, dmd, verb=args.verb)

    print("\nM:done topoAna_daleMatrix4")

    
    # ══════════════════════════════════════════════════════════════
    #  Save results 
    # ══════════════════════════════════════════════════════════════
    outMD ={"dale_conf":dmd}
    outMD["shuffle"] = bool(args.shuffle)
    outMD["provenance"] = {'A-input':args.dataName}
    outMD['methods']={
        'assortativity': info1,
        'jaccard_index': info2,
        'cycle_decay': info3,
        'homology_k1': info4
    }

    outD={'G': G,'A_off': A_off, 'neuron_perm': neuron_perm}
    if args.verb>1: pprint(outMD)
    outF = args.dataName
    outFF = os.path.join(args.outPath, f"{outF}.topoAna.npz")

    write_data_npz(outD, outFF, metaD=outMD)
    print(f"\nSaved: {outFF}")
    print("\n#Summary: dataName, ker_delta, r_assort, jaccard, cycle_decay, hom_k1")
    print("#Values: %s  %.2f  %.6f  %.6f  %.6f  %.6f" % (args.dataName, dmd["placement_ker_delta"], info1["r_assortativity"], info2["mean_jaccard"], info3["cycle_decay"], info4["homology_k1"]))
   
