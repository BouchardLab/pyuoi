#!/usr/bin/env python3

import numpy as np
import os
import argparse
import sys
from toolbox.Util_NumpyIO import read_data_npz
from PlotterEvalFit import Plotter
 
from pprint import pprint

def score_classifier(trueD,fitD,fitType):
    print('\nscore_classifier')
    for ntype in ['exc','inh']:
        tmask=trueD['mask.true.'+ntype]
       # fmask=fitD['mask.%s.%s'%(fitType,ntype)]
        fmask=fitD['mask.lasso.%s'%(ntype)]
        
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
    parser.add_argument('-a',"--ampl_thres", type=float, default=None, help="minima amplitude of valid off-diagonal edge")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="ab", help="Plot types to show: a=structure, b=distributions, c=reconstruction, d=category, d=A-matrix histograms")
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
       
    args = parser.parse_args()
    
    if args.outPath is None:   args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
    print(vars(args))
    

    # Load fit results
    fitFF = os.path.join(args.dataPath, f"{args.dataName}.lasso.npz")
    if os.path.exists(fitFF):            
        fitD, fitMD = read_data_npz(fitFF)
        fitType='lasso'
    else:
        fitFF2=fitFF.replace('lasso','regress')
        assert os.path.exists(fitFF2)
        fitD, fitMD = read_data_npz(fitFF2)
        fitType='regress'

        #... recover some data from lasso fit 
        fitF1=fitMD['fit_lasso']['lassoFit_output_name']
        fitF1F = os.path.join(args.dataPath, fitF1+".lasso.npz")
        fitD1, fitMD1= read_data_npz(fitF1F)
        for xx in[ 'mask.lasso.exc','mask.lasso.inh' ]:
            fitD[xx]=fitD1[xx]
    if args.verb>1: 
        pprint(fitMD); exit(1)

    if 0:
        truthF = fitMD['fit_lasso']['lassoFit_input_name']    
        truthFF = os.path.join(args.dataPath, f"{truthF}.truth.npz")
        if not truthFF or not os.path.exists(truthFF):
            raise FileNotFoundError(f"Truth file not found at {truthFF}")
        
        trueD,trueMD = read_data_npz(truthFF)
   
        # Combine metadata  TMP
        MD = {**fitMD, **trueMD, 'short_name': args.dataName, 'post': vars(args)}
    else:
        MD = {**fitMD,  'short_name': args.dataName, 'post': vars(args)}
        
    if args.ampl_thres!=None:
        maskD['fitL1']=select_eges_from_fitL1(bigD,args.ampl_thres)
        oo1
 
    #1score_classifier(trueD,fitD,fitType)
        
    # Setup plotter
    args.prjName = args.dataName 
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.correl_after_thresh(trueD,fitD,MD, fitType=fitType,figId=1)
                
    if 'b' in args.showPlots:
        plot.slicedA_histos(fitD, MD, fitType=fitType, figId=2)

    if 'c' in args.showPlots:
        plot.residuals(trueD,fitD,MD, fitType=fitType,figId=3)

    if 'd' in args.showPlots:
        plot.compare_eigen(trueD,fitD,MD, fitType=fitType, figId=4)

    if 'e' in args.showPlots:
        plot.experiment_eigen(fitD,MD, fitType=fitType, figId=5)

    plot.display_all()


if __name__ == "__main__":
    main() 
