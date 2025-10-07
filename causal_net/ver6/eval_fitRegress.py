#!/usr/bin/env python3

import numpy as np
import os
import argparse
import sys
from toolbox.Util_NumpyIO import read_data_npz
from PlotterFitEval import Plotter
#from UtilDalePoisson import select_edges_from_fitLasso
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from selectEdges_FDR import  print_table_4_Yao
from UtilSelectFDR import eval_tagged_edges_4_simu, load_auxiliary_plotting_data

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

    if 1:  # patch old data
        #fitMD['data_type']='simDale'
        #fitMD['fit_type']='regress'
        fitD['single_rates']=fitD.pop('firing_rates')

    # Load auxiliary data needed for plotting
    spikeD, trueD, MD = load_auxiliary_plotting_data(fitMD,  args.dataName, args.dataPath)

    if args.verb>1:
        pprint(fitMD); exit(1)
    
    if fitMD['data_type']=='simDale':
        # tmp
        fitD['A_avr']=fitD['A_regress']
        fitD['B_avr']=fitD['B_regress']
        evalD=eval_tagged_edges_4_simu(fitD,trueD)            
        print_table_4_Yao(evalD,fitMD)
        
    # Setup plotter
    args.prjName = args.dataName 
    plot = Plotter(args)
    fitType='regress'

    if 'a' in args.showPlots:
        plot.summary_fitLasso(fitD,MD,figId=1)
 
    if 'xa' in args.showPlots:
        xx1_fix_fig_a
        plot.correl_after_thresh(trueD,fitD,maskD,MD,figId=1)
                
    if 'b' in args.showPlots:
        assert  fitMD['data_type']=='simDale'
        plot.residuals(evalD,MD,figId=2)

        #plot.slicedA_histos(fitD, MD, spikeD, figId=2)
        
    if 'c' in args.showPlots:
        plot.freqSortA_histos(fitD, MD, spikeD, figId=2)

    if 'xc' in args.showPlots:
        plot.residuals(trueD,fitD,maskD,MD,figId=3)

    if 'xd' in args.showPlots:
        plot.compare_eigen(trueD,fitD,MD, figId=4)

    if 'xe' in args.showPlots:
        plot.experiment_eigen(fitD,MD, figId=5)

    plot.display_all()


if __name__ == "__main__":
    main() 
