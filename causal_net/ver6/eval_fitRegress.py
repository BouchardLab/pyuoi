#!/usr/bin/env python3
"""
Evaluation and visualization tool for regression model fitting results.

This script provides evaluation capabilities for regression-based connectivity
analysis, focusing on edge selection and network reconstruction quality.
Main functionality includes:
- Loading regression fit results and applying edge selection thresholds
- Statistical analysis of reconstructed network properties
- Visualization of connectivity patterns and edge distributions
- Performance metrics computation for network reconstruction

Complements the LASSO-based analysis by providing alternative regression
approaches for neural connectivity inference.
"""

import numpy as np
import os
import argparse
import sys
from toolbox.Util_NumpyIO import read_data_npz
from PlotterFitEval import Plotter
from UtilDalePoisson import select_edges_from_fitLasso
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz

from pprint import pprint

#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from fit_poisson.py")
    parser.add_argument("--dataName", type=str, default='dale_M120_3M', help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="path to input and output files")
    
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="ab", help="Plot types to show")
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
       
    args = parser.parse_args()
    np.set_printoptions(precision=3)
  
    if args.outPath is None:   args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
    print(vars(args))
   
    # Load fit results
    fitFF = os.path.join(args.dataPath, f"{args.dataName}.regressFit.npz")
    fitD, fitMD = read_data_npz(fitFF)

    if 0:  # patch old data
        fitMD['data_type']='simDale'
        fitMD['fit_type']='regress'

    if args.verb>1: 
        pprint(fitMD); exit(1)

    maskF=fitMD['fit_regress']['regressFit_input_name']
    maskFF = os.path.join(args.dataPath, maskF+".edgeMask.npz")
    maskD, maskMD = read_data_npz(maskFF)
       
    if fitMD['data_type']=='simDale':
        truthF = fitMD['fit_lasso']['lassoFit_input_name']    
        truthFF = os.path.join(args.dataPath, f"{truthF}.simTruth.npz")
        trueD,trueMD = read_data_npz(truthFF)
        
        # Load spike data for frequency sorting
        spikesFF = os.path.join(args.dataPath, f"{truthF}.spikes.npz")
        spikeD, spikeMD = read_data_npz(spikesFF)
        
        # Combine metadata   just for plotter
        MD = {**fitMD, **trueMD, 'short_name': args.dataName} #, 'post': vars(args)}
    else:
        MD = {**fitMD,  'short_name': args.dataName} #, 'post': vars(args)}
        # For non-simDale data, we don't have spike data, so create a minimal spikeD
        spikeD = None

    MD.update(maskMD)
    
    # Setup plotter
    args.prjName = args.dataName 
    plot = Plotter(args)
    fitType='lasso'
    if 'a' in args.showPlots:
        plot.correl_after_thresh(trueD,fitD,maskD,MD,figId=1)
                
    if 'b' in args.showPlots:
        plot.slicedA_histos(fitD, MD, spikeD, figId=2)

    if 'c' in args.showPlots:
        plot.residuals(trueD,fitD,maskD,MD,figId=3)

    if 'd' in args.showPlots:
        plot.compare_eigen(trueD,fitD,MD, figId=4)

    if 'e' in args.showPlots:
        plot.experiment_eigen(fitD,MD, figId=5)

    plot.display_all()


if __name__ == "__main__":
    main() 
