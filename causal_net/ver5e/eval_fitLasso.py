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


def print_scores(trueD,fitD,maskD):  # saves no output!
    print('\nscore_classifier')
    fmask=maskD['mask.lasso.exist']
    for ntype in ['exc','inh']:
        tmask=trueD['mask.true.'+ntype]
          
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
    parser.add_argument('-a',"--ampl_thres", type=float, default=0.10, help="minima amplitude of valid off-diagonal edge")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="ab", help="Plot types to show: a=structure, b=distributions, c=reconstruction, d=category, d=A-matrix histograms")
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
    fitType='lasso'
    
    if 0:  # patch old data
        fitMD['type']='simDale'

    #pprint(fitMD)
   
    if args.verb>1: 
        pprint(fitMD); exit(1)

    maskF,maskMD=select_edges_from_fitLasso(fitD,args.ampl_thres)
    maskD={}
    for xx in maskF:
            maskD['mask.lasso.'+xx]=maskF[xx]

    if fitMD['type']=='simDale':
        truthF = fitMD['fit_lasso']['lassoFit_input_name']    
        truthFF = os.path.join(args.dataPath, f"{truthF}.simTruth.npz")
        trueD,trueMD = read_data_npz(truthFF)

        print_scores(trueD,fitD,maskD)
        
        # Combine metadata  TMP
        MD = {**fitMD, **trueMD, 'short_name': args.dataName} #, 'post': vars(args)}
    else:
        MD = {**fitMD,  'short_name': args.dataName} #, 'post': vars(args)}

    MD.update(maskMD)
        
    outFt = os.path.join(args.outPath, args.dataName + '.edgeMask.npz')
    write_data_npz(maskD, outFt, metaD=maskMD)

    # Setup plotter
    args.prjName = args.dataName 
    plot = Plotter(args)
    fitType='lasso'
    if 'a' in args.showPlots:
        plot.correl_after_thresh(trueD,fitD,maskD,MD, fitType=fitType,figId=1)
                
    if 'b' in args.showPlots:
        plot.slicedA_histos(fitD, MD, fitType=fitType, figId=2)

    if 'c' in args.showPlots:
        plot.residuals(trueD,fitD,maskD,MD, fitType=fitType,figId=3)

    if 'd' in args.showPlots:
        plot.compare_eigen(trueD,fitD,MD, fitType=fitType, figId=4)

    if 'e' in args.showPlots:
        plot.experiment_eigen(fitD,MD, fitType=fitType, figId=5)

    plot.display_all()


if __name__ == "__main__":
    main() 
