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
    ./eval_fitLasso.py --dataName mydata --dataPath /path/to/data/ -p ab
"""

import numpy as np
import os
import argparse
import sys
from toolbox.Util_NumpyIO import read_data_npz
from PlotterFitEval import Plotter
#from UtilDalePoisson import select_edges_from_fitLasso
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz

from pprint import pprint

#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from fit_poisson.py")
    parser.add_argument("--dataName", type=str, default='dale_M120_3M', help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="Path to the data directory")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="f", help="Plot types to show: a=structure, b=distributions, c=reconstruction, d=category, d=A-matrix histograms")
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
       
    args = parser.parse_args()
    np.set_printoptions(precision=3)
    
    if args.outPath is None:   args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
    print(vars(args))
    
    # Load fit results
    fitFF = os.path.join(args.dataPath, f"{args.dataName}.lassoFit.npz")
    fitD, fitMD = read_data_npz(fitFF)
      
    if 0:  # patch old data
        #pprint(fitMD)
        #fitMD['fit_type']='lasso'
        fitMD['fit_lasso']['num_epochs']=fitMD['fit_lasso']['n_epochs']
    #pprint(fitMD)
   
    if args.verb>1: 
        pprint(fitMD); exit(1)

    #1maskD,maskMD=select_edges_from_fitLasso(fitD,args.ampl_thres)
    #1maskMD['fit_lasso']=fitMD['fit_lasso']

    # Load spike data for frequency sorting
    spikeF = fitMD['fit_lasso']['lassoFit_input_name']    
    spikesFF = os.path.join(args.dataPath, f"{spikeF}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF)
    #pprint(spikeMD)
    
    if fitMD['data_type']=='simDale': 
        truthFF = os.path.join(args.dataPath, f"{spikeF}.simTruth.npz")
        trueD,trueMD = read_data_npz(truthFF)    
        # Combine metadata   just for plotter
        MD = {**fitMD, **trueMD, 'short_name': args.dataName} #, 'post': vars(args)}
    else:
        MD = {**fitMD,  'short_name': args.dataName} #, 'post': vars(args)}
    
    # Setup plotter
    args.prjName = args.dataName 
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_fitLasso(fitD,MD,figId=1)
                         
    if 'c' in args.showPlots:
        plot.freqSortA_histos(fitD, MD, spikeD, figId=2)

    plot.display_all()

if __name__ == "__main__":
    main() 
