#!/usr/bin/env python3
"""

"""

import numpy as np
import os
import argparse
import sys
from toolbox.Util_NumpyIO import read_data_npz
from PlotterFitEval2 import Plotter
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz

from pprint import pprint

#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from fit_poisson.py")
    parser.add_argument("--dataName", type=str, default='dale_M120_3M', help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="Path to the data directory")
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="a b", help="Plot types to show: a=summary, b=correlations, c=correlations_for_kris")
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
       
    args = parser.parse_args()
    np.set_printoptions(precision=3)
    
    if args.outPath is None:   args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
    print(vars(args))
    
    # Load fit results
    fitFF = os.path.join(args.dataPath, f"{args.dataName}.uoiFdr.npz")
    fitD, fitMD = read_data_npz(fitFF)
    MD={ **fitMD,  'short_name': args.dataName}
        
    if 0:  # patch old data
        #pprint(fitMD)
        fitMD['fit_type']='uoiFdr'
    #pprint(fitMD)
   
    if args.verb>1: 
        pprint(fitMD); exit(1)
   
    # Load spike data for frequency sorting
    spikeF = fitMD['fit_uoi']['fit_input_name']
    inpPath= fitMD['fit_uoi']['fit_input_path']
    spikesFF = os.path.join(inpPath, f"{spikeF}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF)
    #pprint(spikeMD)
    
    if fitMD['spike_data']['data_type']=='simDale': 
        truthFF = os.path.join(inpPath, f"{spikeF}.simTruth.npz")
        trueD,trueMD = read_data_npz(truthFF)
        #pprint(trueMD)
        for xx in ['short_name']:
            trueMD.pop(xx)
        # Combine metadata   just for plotter
        MD.update( **trueMD)
    
    # Setup plotter
    args.prjName = args.dataName 
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_fitUoI(fitD,trueD,MD,figId=1)
    if 'b' in args.showPlots:
        plot.correlations_fitUoI(fitD,trueD,MD,figId=2)
    if 'c' in args.showPlots:
        plot.correlation_for_kris(fitD,trueD,MD,figId=3)
    if 'd' in args.showPlots:
        plot.eigenvalues_fitUoI(fitD,trueD,MD,figId=4)
    if 'e' in args.showPlots:
        plot.pseudospectra_fitUoI(fitD,trueD,MD,figId=5)
   
    plot.display_all()

if __name__ == "__main__":
    main() 
