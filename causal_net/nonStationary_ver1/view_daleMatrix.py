#!/usr/bin/env python3
"""
Visualization tool for simulated Dale Poisson network data.

Loads simulation output (simTruth.npz and spikes.npz) produced by
gen_daleMatrices.py and plots analysis figures for a selected spectral
radius index (--idxR).  All arrays use natural neuron indexing
(first num_excite neurons are excitatory, remainder inhibitory).

Available plots (-p flag):
  a  Dale connectivity matrix (color-coded) + eigenvalue scatter
     with spectral-radius circle
  b  Weight histogram, firing-rate histogram, outgoing-edge count
     per neuron, and per-neuron firing-rate bar chart
  d  B_idle vs firing rate / SNR scatter, plus excitatory and
     inhibitory rate histograms
  e  Pseudospectral contour plot with eigenvalue overlay

Usage:
    ./view_daleMatrix.py --dataName daleN100_9fbe7f -i 0 -p a b d e
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
from pprint import pprint
import numpy as np
from PlotterDaleMatrix import Plotter
from toolbox.Util_NumpyIO import read_data_npz
import argparse

#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser(description="Visualize simulated Dale Poisson network data")
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a b', nargs='+',help="abc-string listing shown plots: a=Dale_matrix_and_eigen, b=histo_weights_rates, d=rates_study, e=pseudospectra")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--basePath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for input data")
    parser.add_argument("--dataName",  default='daleN150_448b86',help='simulated Dale network base name')
    #parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to basePath)")
    parser.add_argument('-i', '--idxR', type=int, default=0, help="Index into spect_radius list, selects which R to plot")
    
    args = parser.parse_args()
    
    # make arguments more flexible
    args.inpPath = os.path.join(args.basePath,'truthDale')
    args.outPath = os.path.join(args.basePath,'plots')
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.basePath)
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
    truthFF = os.path.join(args.inpPath, f"{args.dataName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb>0)
    if args.verb>1: 
        print("\nSimulation Truth Metadata:")
        pprint(trueMD)
    
    # Load spike data (generated spike counts and rates)
    spikesFF = os.path.join(args.inpPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb>0)
    if args.verb>1:
        print("\nSpike Data Metadata:")
        pprint(spikeMD)
    
    # Merge metadata for plotting
    trueMD['short_name'] = args.dataName
    
    # Select spectral radius slice
    ir = args.idxR
    spect_radii = trueMD['dale_conf']['spect_radius']
    assert ir < len(spect_radii), f"idxR={ir} out of range, only {len(spect_radii)} radii available"
    R_sel = spect_radii[ir]
    print(f"\nSelected spectral radius [{ir}]: R={R_sel:.3f}  (out of {spect_radii})")
    
    # Slice stacked arrays along axis 0 for the selected R
    trueD_r = { 'A_true': trueD['A_true'][ir], 'B_true': trueD['B_true'][ir], 'E_true': trueD['E_true'] }
    spikeD_r = { k: spikeD[k][ir] for k in spikeD }
    trueMD['sel_spect_radius'] = R_sel
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=args.dataName+'_view%d'%ir
    plot=Plotter(args)
    
    if 'a' in args.showPlots:
        plot.Dale_matrix_and_eigen(trueD_r['A_true'],trueMD,trueD_r,figId=1)
    
    if 'b' in args.showPlots:
        plot.histo_weights_rates(trueD_r,spikeD_r,trueMD,figId=2)
 
    if 'c' in args.showPlots:
        plot.rates_study(trueD_r,spikeD_r,trueMD,figId=3)

    if 'd' in args.showPlots:
        plot.Dale_matrix_pseudospectra(trueD_r['A_true'],trueMD,trueD_r,figId=4)
 
    plot.display_all()
    print('M:done - view_dalePoisson completed successfully!')
