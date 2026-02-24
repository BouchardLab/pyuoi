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

######################### 
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from fit_poisson.py")
    parser.add_argument("--dataName", type=str, default='dale_M120_3M', help="Base name for the dataset")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input/output data")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="f", help="Plot types to show: a=structure, b=edge detection quality, c=reconstruction, d=category, d=A-matrix histograms")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
       
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
      
    if 0:  # patch old data
        #pprint(fitMD)
        #fitMD['fit_type']='lasso'
        fitMD['fit_lasso']['num_epochs']=fitMD['fit_lasso']['n_epochs']
    #pprint(fitMD)
   
    if args.verb>1: 
        pprint(fitMD); exit(1)

    # Load spike data for frequency sorting
    spikeF = fitMD['fit_lasso']['lassoFit_input_name']
    inpPath2= fitMD['fit_lasso']['lassoFit_input_path']
    spikesFF = os.path.join(inpPath2, f"{spikeF}.spikes.npz")
 
    spikeD, spikeMD = read_data_npz(spikesFF)
    #pprint(spikeMD)
    # Combine metadata   just for plotter
    MD = {**fitMD,  'short_name': args.dataName} 
    
    if 'simDale' in fitMD['data_type']:
        truthPath=inpPath2
        truthF=spikeF
    if 'simPrism' in fitMD['data_type']:
        truthPath= os.path.join(args.basePath, 'truthDale/')
        truthF=spikeMD['input_truth_name']
        
    truthFF = os.path.join(truthPath, f"{truthF}.simTruth.npz")    
    trueD,trueMD = read_data_npz(truthFF)       
    MD.update(  trueMD )
    MD['E_true']=trueD['E_true']
    MD['A_true']=trueD['A_true']

    #- - - - - - - - - - - - - - - - - Setup plotter - - - - - - - - - - - - - - - - 
    args.prjName = args.dataName 
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_fitLasso(fitD,MD,figId=1)

    if 'b' in args.showPlots:
        plot.edges_fitLasso(fitD,MD,minW=0.,figId=2)
    
    if 'c' in args.showPlots:
        plot.freqSortA_histos(fitD, MD, spikeD, figId=3)

    if 'd' in args.showPlots:
        plot.edges_fitLasso(fitD,MD,minW=0.1,figId=2)

    plot.display_all()

if __name__ == "__main__":
    main() 
