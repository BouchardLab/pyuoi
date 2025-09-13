#!/usr/bin/env python3

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
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="Path to the data directory")
    parser.add_argument('-A',"--ampl_thres", type=float, default=[0.10],nargs='+', help=" inh< tht0, exct>th1 of accepted off-diagonal edge")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="ab", help="Plot types to show: a=structure, b=distributions, c=reconstruction, d=category, d=A-matrix histograms")
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
       
    args = parser.parse_args()
    np.set_printoptions(precision=3)

    if len(args.ampl_thres)==1:
        args.ampl_thres=[-args.ampl_thres[0],args.ampl_thres[0]]
    assert len(args.ampl_thres)==2
    assert args.ampl_thres[0]*args.ampl_thres[1] <0
    
    if args.outPath is None:   args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
    print(vars(args))
   
    # Load fit results
    fitFF = os.path.join(args.dataPath, f"{args.dataName}.lassoFit.npz")
    fitD, fitMD = read_data_npz(fitFF)
      
    if 0:  # patch old data
        fitMD['data_type']='simDale'
        #fitMD['fit_type']='lasso'
    #pprint(fitMD)
   
    if args.verb>1: 
        pprint(fitMD); exit(1)

    maskD,maskMD=select_edges_from_fitLasso(fitD,args.ampl_thres)
    maskMD['fit_lasso']=fitMD['fit_lasso']

    # Load spike data for frequency sorting
    spikeF = fitMD['fit_lasso']['lassoFit_input_name']    
    spikesFF = os.path.join(args.dataPath, f"{spikeF}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF)

    if fitMD['data_type']=='simDale':
        #truthF = fitMD['fit_lasso']['lassoFit_input_name']    
        truthFF = os.path.join(args.dataPath, f"{spikeF}.simTruth.npz")
        trueD,trueMD = read_data_npz(truthFF)    
        # Combine metadata   just for plotter
        MD = {**fitMD, **trueMD, 'short_name': args.dataName} #, 'post': vars(args)}
    else:
        MD = {**fitMD,  'short_name': args.dataName} #, 'post': vars(args)}
    
    MD.update(maskMD)
        
    outFt = os.path.join(args.outPath, args.dataName + '.edgeMask.npz')
    write_data_npz(maskD, outFt, metaD=maskMD)

    # Setup plotter
    args.prjName = args.dataName 
    plot = Plotter(args)
  
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
        
    if 'f' in args.showPlots:
        plot.freqSortA_histos(fitD, MD, spikeD, figId=2)

    plot.display_all()


if __name__ == "__main__":
    main() 
