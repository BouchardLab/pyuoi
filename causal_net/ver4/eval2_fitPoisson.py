#!/usr/bin/env python3
"""
Evaluation and plotting script for analyzing results from fit_poisson.py.

This script:
1. Loads the learned A & B matrices from fit_poisson.py

Usage:
    python eval2_fitPoisson.py --dataName <name> --dataPath <path> --showPlots <plot_types> --target_sparsity <sparsity>

Example:
    python eval1_fitPoisson.py --dataName test_dale --dataPath out/ --showPlots abc --target_sparsity 0.8
"""

import numpy as np
import os
import argparse
import sys
from PlotterFitPoissonV2 import Plotter
from UtilFitPoisson import  analyze_edge_detection, save_evaluation_results
from pprint import pprint

def load_fit_results(dataName, dataPath):
    """
    Load all results from fit_poisson.py output files.
    
    Args:
        dataName: Base name for the dataset
        dataPath: Path to the data directory
        
    Returns:
        dict: Dictionary containing all loaded data
    """
    print(f"Loading fit results for {dataName} from {dataPath}")
    
    # Construct file paths
    struct_file = os.path.join(dataPath, f"{dataName}.struct.npy")
    truth_file = os.path.join(dataPath, f"{dataName}.truth.npz")
    #spikes_file = os.path.join(dataPath, f"{dataName}.spikes.npz")
      
    print(f"Loading structure results from {struct_file}")
    struct_data = np.load(struct_file, allow_pickle=True).item()
    A_stage1 = struct_data['A_stage1']
    B_stage1 = struct_data['B_stage1']
    train_losses = struct_data['train_losses']
    val_losses = struct_data['val_losses']
    firing_rates = struct_data['firing_rates']
    print(sorted(struct_data))
    fit_conf = struct_data['args']
    #pprint(fit_conf);dd2

    print(f"Loaded structure data: A_stage1 shape={A_stage1.shape}")
    print(f"Training losses: {len(train_losses)} epochs")
    
    # Load ground truth if available
    print(f"Loading ground truth from {truth_file}")
    truth_data = np.load(truth_file, allow_pickle=True)
    A_true = truth_data['A']
    B_true = truth_data['B_intercept']
    dale_conf = truth_data['conf'].item()
    evol_conf = truth_data['evol_conf'].item()
   
    #pprint(evol_conf);dd
    
    bigD={
        'A_true': A_true,
        'B_true': B_true,
        'A_fit': A_stage1,
        'B_fit': B_stage1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'firing_rates': firing_rates}

    num_epochs = len(train_losses)
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
   
    md={'dale':dale_conf,
        'evol':evol_conf,
        'fit':fit_conf,
        'short_name': dataName,
        
        #'initial_lr': initial_lr,
        #'train_time_min': train_time_min,
        'num_epochs': num_epochs,
        'final_loss': final_val_loss,  # Add final validation loss
        'final_train_loss': final_train_loss,  # Add final training loss
        'final_val_loss': final_val_loss  # Add final validation loss (alternative key)
    }
    
    
    
    return bigD,md

def geom_edge_mask(md):
    dale_conf=md['dale']
    pprint(dale_conf)
    Nn=dale_conf['num_neurons']
    Ne=dale_conf['num_excite']
    Ni=Nn-Ne

    # Create diagonal mask
    diag_mask = np.eye(Nn, dtype=bool)
    # exc_mask: first Ne rows, all columns, except diagonal
    exc_mask = np.zeros((Nn, Nn), dtype=bool)
    exc_mask[:Ne, :] = True
    exc_mask = exc_mask & (~diag_mask)  # remove diagonal

    # inh_mask: next Ni rows, all columns, except diagonal
    inh_mask = np.zeros((Nn, Nn), dtype=bool)
    inh_mask[Ne:, :] = True
    inh_mask = inh_mask & (~diag_mask)  # remove diagonal

    # create 1d masks for exc & inh
    exc_1d= np.zeros((Nn), dtype=bool)
    exc_1d[:Ne] = True
    inh_1d= np.zeros((Nn), dtype=bool)
    inh_1d[Ne:] = True
    
    maskG={'diag':diag_mask, 'exc':exc_mask,'inh':inh_mask,'exc_1d':exc_1d,'inh_1d':inh_1d}
    maskD={'geom':maskG}
    return maskD

def true_edge_mask(maskD, A_true):
    print('\ntrue_edge_mask')
    maskD['true']=maskT={}
    maskG=maskD['geom']
    A_abs = np.abs(A_true)
    for ntype in ['exc','inh']:
        gmask=maskG[ntype]
        tmask = gmask & (A_abs>1e-8)
        nGeom=np.sum(gmask)
        nTrue=np.sum(tmask)
        print('true mask',ntype,nGeom,nTrue)
        maskT[ntype]=tmask

def fit_edge_mask(maskD, bigD,density=0.2):
    print('\nfit_edge_mask target density=%.2f'%density)
    maskD['fit']=maskF={}
    maskG=maskD['geom']
    pmask=maskG['diag']
    A_fit=bigD['A_fit']
    A_abs = np.abs(A_fit)
    for ntype in ['exc','inh']:
        gmask=maskG[ntype]
        A_sel=A_abs[gmask]
        assert len(A_sel)>0
        # Find threshold for top 'density' fraction
        thresh = np.percentile(A_sel, 100 * (1 - density))
        # fmask: True where A_abs > thresh *and* in gmask
        # I want fmask to be the same shape as your original mask array
        fmask = np.zeros_like(gmask, dtype=bool)
        fmask[gmask] = A_sel > thresh
        maskF[ntype] = fmask
        nGeom=np.sum(gmask)
        nFit=np.sum(fmask)
        print('fit mask',ntype,nGeom,nFit,'thres=%.3f'%thresh)
        # pmask is True wherever maskG['diag'] OR fmask is True
        pmask = pmask | fmask  # element-wise logical OR
    maskF['pass']=pmask
    # Set to zero where pmask is False, same shape as A_fit
    A_pass = np.where(pmask, A_fit, 0)
    print('ss1',A_fit.shape, A_pass.shape)
    bigD['A_pass']=A_pass
    

def score_classifier(maskD):
    print('\nscore_classifier')        
    maskT=maskD['true']
    maskF=maskD['fit']
    for ntype in ['exc','inh']:
        tmask=maskT[ntype]
        fmask=maskF[ntype]
        
        FP = np.sum(~tmask &  fmask)  # False Positive: predicted True, actually False
        TP = np.sum( tmask &  fmask)  # True Positive: predicted True, actually True
        TN = np.sum(~tmask & ~fmask)  # True Negative: predicted False, actually False
        FN = np.sum( tmask & ~fmask)  # False Negative: predicted False, actually True

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        recal = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2*prec*recal/(prec+recal) if (prec + recal) > 0 else 0

        nTrue = np.sum(tmask)
        nFit = np.sum(fmask)
        print('score', ntype, 'nT=%d nF=%d' % (nTrue, nFit),
              'TP=%d FP=%d TN=%d FN=%d prec=%.3f rec=%.3f F1=%.3f' % (
                  TP, FP, TN, FN, prec, recal, F1))



#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from fit_poisson.py")
    parser.add_argument("--dataName", type=str, default='dale_M120_3M', help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="out/", help="Path to the data directory")
    parser.add_argument('-d',"--target_density", type=float, default=None, help="Target density level (0.2 = 80% zeros), or use simu conf")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="ab", help="Plot types to show: a=structure, b=distributions, c=reconstruction, d=category, d=A-matrix histograms")
    parser.add_argument("--outPath", type=str, default="out/", help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level")
       
    args = parser.parse_args()
    
    # Set output path
    if args.outPath is None:   args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
    print(vars(args))
    print("")

    
    # Load all fit results
    bigD,MD = load_fit_results(args.dataName, args.dataPath)
    if args.target_density==None:
        args.target_density=MD['dale']['p']
        
    MD['post']=vars(args)
    
    maskD=geom_edge_mask(MD)
    true_edge_mask(maskD,bigD['A_true'])
    fit_edge_mask(maskD,bigD,args.target_density)
    score_classifier(maskD)
        
    # Setup plotter
    args.prjName = args.dataName + '_struct'
    plot = Plotter(args)
    
    # Generate requested plots
    print(f"\n=== Generating Plots ===")
    
    if 'a' in args.showPlots:
        plot.correl_after_thresh(bigD, maskD,MD, figId=1)
        
    if 'b' in args.showPlots:
        plot.daleA_and_eigen(bigD, maskD,MD, figId=2)
        
    if 'd' in args.showPlots:
        plot.plot_slicedA_histos(bigD, maskD,MD, figId=3)

    # Display all plots
    plot.display_all()
    
    print("\n=== Evaluation Complete ===")
    print(f"Plots saved to: {args.outPath}")
    

if __name__ == "__main__":
    main() 
