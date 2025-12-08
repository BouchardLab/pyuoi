#!/usr/bin/env python3
"""
Visualization tool for simulated Dale Poisson network data.

This script provides comprehensive visualization of simulated Dale's principle
neural networks with Poisson spiking dynamics. It loads previously generated
simulation data and produces various analysis plots.

Main functionality includes:
- Dale connectivity matrix visualization (natural and frequency-sorted order)
- Eigenvalue analysis showing network stability properties
- Weight and firing rate distribution histograms
- Neuron type (excitatory/inhibitory) analysis
- Comparison of different network orderings

Used for post-simulation analysis and validation of Dale network properties
without re-running the full simulation.

Usage:
    ./view_dalePoisson.py --dataName daleM150_448b86 -p a b c
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
from pprint import pprint
import numpy as np
from PlotterSimPoisson import Plotter
from toolbox.Util_NumpyIO import read_data_npz
import argparse

#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser(description="Visualize simulated Dale Poisson network data")
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a b', nargs='+',help="abc-string listing shown plots: a=Dale_matrix_and_eigen, b=histo_weights_rates, c=histo_weights_rates_byFreq, d=rates_study")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--dataPath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for input data")
    parser.add_argument("--dataName",  default='daleM150_448b86',help='simulated Dale network base name')
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    
    args = parser.parse_args()
    
    # make arguments more flexible
    if args.outPath is None:   args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    return args

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
    
    # Load simulation truth data (Dale matrices, biases, etc.)
    truthFF = os.path.join(args.dataPath, f"{args.dataName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb>0)
    if args.verb>1: 
        print("\nSimulation Truth Metadata:")
        pprint(trueMD)
    
    # Load spike data (generated spike counts and rates)
    spikesFF = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb>0)
    if args.verb>1:
        print("\nSpike Data Metadata:")
        pprint(spikeMD)
    
    # Merge metadata for plotting
    trueMD['short_name'] = args.dataName
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=args.dataName+'_view'
    plot=Plotter(args)
    
    if 'a' in args.showPlots:
        plot.Dale_matrix_and_eigen(trueD['A_true'],trueMD,trueD,figId=1)
    
    if 'b' in args.showPlots:
        plot.histo_weights_rates(trueD,spikeD,trueMD,figId=2)
 
    if 'c' in args.showPlots:
        plot.histo_weights_rates(trueD,spikeD,trueMD,byFreq=True,figId=3)
  
    if 'd' in args.showPlots:
        plot.rates_study(trueD,spikeD,trueMD,figId=4)

    if 'e' in args.showPlots:
        plot.Dale_matrix_pseudospectra(trueD['A_true'],trueMD,trueD,figId=5)
 
    plot.display_all()
    print('M:done - view_dalePoisson completed successfully!')

