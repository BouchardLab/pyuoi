#!/usr/bin/env python3
"""
Evaluation and visualization tool for LASSO Poisson model fitting results.

This script loads fitted LASSO connectivity matrices from Poisson GLM training
and provides comprehensive evaluation through statistical analysis and plotting.
Main functionality includes:
- Loading fitted model parameters (A, B matrices) from .lassoFit.npz files
- Computing network connectivity statistics and sparsity metrics
- Generating various visualizations (structure plots, distributions, reconstructions)
- Optionally comparing against ground truth for simulated data

Usage:
    ./eval_fitLasso.py --dataName mydata --basePath /path/to/data/ -p ab
"""

import numpy as np
import os
import argparse
import sys
from toolbox.Util_NumpyIO import read_data_npz
from PlotterLassoFitEval import Plotter

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz

from pprint import pprint


def get_offdiag_triplets(A, is_pos=True, min_w=None):
    """Return off-diagonal [i, j, value] triplets filtered by sign and abs(value)>=min_w."""
    A = np.asarray(A)
    assert A.ndim == 2 and A.shape[0] == A.shape[1], f"Expected square 2D matrix, got {A.shape}"
    assert min_w is not None, "min_w must be provided"
    offdiag = ~np.eye(A.shape[0], dtype=bool)
    sign_mask = (A > 0) if is_pos else (A < 0)
    mag_mask = np.abs(A) >= float(min_w)
    mask = offdiag & sign_mask & mag_mask
    i_idx, j_idx = np.where(mask)
    vals = A[i_idx, j_idx]
    return np.column_stack([i_idx, j_idx, vals])


def compare_triplets(rec, truth):
    """Return [TP, FP, FN] for triplets; TP carries both reconstructed and true values."""
    rec_dict = {(int(r[0]), int(r[1])): r[2] for r in rec}
    tru_dict = {(int(r[0]), int(r[1])): r[2] for r in truth}

    rec_idx = set(rec_dict.keys())
    tru_idx = set(tru_dict.keys())

    tp_idx = rec_idx & tru_idx
    fp_idx = rec_idx - tru_idx
    fn_idx = tru_idx - rec_idx

    tp = np.array([[i, j, rec_dict[(i, j)], tru_dict[(i, j)]] for i, j in sorted(tp_idx)])
    fp = np.array([[i, j, rec_dict[(i, j)]] for i, j in sorted(fp_idx)])
    fn = np.array([[i, j, tru_dict[(i, j)]] for i, j in sorted(fn_idx)])

    if len(tp) == 0:
        tp = np.empty((0, 4))
    if len(fp) == 0:
        fp = np.empty((0, 3))
    if len(fn) == 0:
        fn = np.empty((0, 3))

    return [tp, fp, fn]


def build_residual_eval_data(fitD, trueD, minW, verb=1):
    """Build evalD expected by PlotterLassoFitEval.residuals()."""
    assert minW is not None, "minW must be provided"
    A_fit = np.asarray(fitD['A_lasso'])
    A_true = np.asarray(trueD['A_true'])
    if A_true.ndim == 3:
        A_true = A_true[0]

    assert A_fit.ndim == 2 and A_true.ndim == 2, "A_lasso/A_true must be 2D matrices"
    if A_fit.shape != A_true.shape:
        n = min(A_fit.shape[0], A_true.shape[0])
        if verb:
            print(f"WARNING: A shape mismatch fit={A_fit.shape}, true={A_true.shape}; using top-left {n}x{n}.")
        A_fit = A_fit[:n, :n]
        A_true = A_true[:n, :n]

    EposT = get_offdiag_triplets(A_true, True, min_w=minW)
    EnegT = get_offdiag_triplets(A_true, False, min_w=minW)
    EposR = get_offdiag_triplets(A_fit, True, min_w=minW)
    EnegR = get_offdiag_triplets(A_fit, False, min_w=minW)

    if verb:
        print(f"Residual eval edges (|w|>={minW:g}): true pos/neg={EposT.shape[0]}/{EnegT.shape[0]}, fit pos/neg={EposR.shape[0]}/{EnegR.shape[0]}")

    evalD = {
        'pos': compare_triplets(EposR, EposT),
        'neg': compare_triplets(EnegR, EnegT),
        'diag': np.column_stack([np.diag(A_fit), np.diag(A_true)]),
        'minW': float(minW),
    }

    B_fit = fitD.get('B_lasso')
    B_true = trueD.get('B_true')
    if B_fit is not None and B_true is not None:
        B_fit = np.asarray(B_fit).reshape(-1)
        B_true = np.asarray(B_true)
        if B_true.ndim == 2:
            if verb and B_true.shape[0] > 1:
                print(f"INFO: B_true has {B_true.shape[0]} rows; using row 0 for residual plot.")
            B_true = B_true[0]
        B_true = np.asarray(B_true).reshape(-1)
        nB = min(B_fit.size, B_true.size)
        if B_fit.size != B_true.size and verb:
            print(f"WARNING: B length mismatch fit={B_fit.size}, true={B_true.size}; using first {nB}.")
        evalD['bterm'] = np.column_stack([B_fit[:nB], B_true[:nB]])

    return evalD

######################### 
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from fit_poisson.py")
    parser.add_argument("--dataName", type=str, default='dale_M120_3M', help="Base name for the dataset")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input/output data")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="f", help="Plot types to show: a=summary, b=edge quality (minW=0), c=A-matrix histograms, d=edge quality (minW=arg), e=residuals")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
    parser.add_argument("--minW", type=float, default=0.01, help="Threshold for A-matrix eval, not for fitting")
       
    args = parser.parse_args()
    args.inpPath = os.path.join(args.basePath, 'lassoFdrFit')
    args.outPath = os.path.join(args.basePath, 'plots')
    os.makedirs(args.outPath, exist_ok=True)
    np.set_printoptions(precision=3)
    args.showPlots=''.join(args.showPlots)
    print(vars(args))
    
    # Load fit results
    fitFF = os.path.join(args.inpPath, f"{args.dataName}.lassoFit.npz")
    fitD, fitMD = read_data_npz(fitFF)
        
    if args.verb>1: 
        pprint(fitMD); exit(1)

    # Load spike data for frequency sorting
    spikeF = fitMD['provenance']['state_transition_file']    
    inpPath2=os.path.join(args.basePath, 'spikesData')
    spikesFF = os.path.join(inpPath2, f"{spikeF}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF)
    
    #pprint(spikeMD)
    MD = {**fitMD,  'short_name': args.dataName} 
    
    if 'simDale' in fitMD['data_type']:
        truthPath=inpPath2
        truthF=spikeF
    if 'simPrism' in fitMD['data_type']:
        truthPath= os.path.join(args.basePath, 'truthDale/')
        truthF=fitMD['provenance']['state_model_file']
        
    truthFF = os.path.join(truthPath, f"{truthF}.simTruth.npz")    
    trueD,trueMD = read_data_npz(truthFF)       
    #MD.update(  trueMD )
    fitD['E_true']=trueD['E_true']
    fitD['A_true']=trueD['A_true']

    #- - - - - - - - - - - - - - - - - Setup plotter - - - - - - - - - - - - - - - - 
    args.prjName = args.dataName 
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_fitLasso(fitD,MD,figId=1)

    if 'b' in args.showPlots:
        plot.edges_fitLasso(fitD,MD,minW=0.,figId=2)
    
    if 'c' in args.showPlots:
        plot.A_histos(fitD, MD, spikeD, figId=3)

    if 'd' in args.showPlots:
        plot.edges_fitLasso(fitD,MD,minW=args.minW,figId=4)

    if 'e' in args.showPlots:
        evalD = build_residual_eval_data(fitD, trueD, minW=args.minW, verb=args.verb)
        plot.residuals(evalD, MD, figId=5)

    plot.display_all()

if __name__ == "__main__":
    main() 
