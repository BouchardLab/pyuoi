#!/usr/bin/env python3
"""
Visualization tool for simulated Dale Poisson network data.

Loads simulation output (simTruth.npz and spikes.npz) produced by
gen_daleMatrices3c.py and plots analysis figures for a selected state
index (--idxState).  All arrays use natural neuron indexing
(first num_excite source columns are excitatory, remainder inhibitory).

Available plots (-p flag):
  a  Dale connectivity matrix, eigenvalue scatter with spectral-radius
     circle, and node-location map colored by signed firing rate
  b  Weight histogram, firing-rate histogram, outgoing-edge count
     per neuron, and per-neuron firing-rate bar chart
  d  B_idle vs firing rate / SNR scatter, plus excitatory and
     inhibitory rate histograms
  e  Pseudospectral contour plot with eigenvalue overlay

Usage:
    ./view_daleMatrix3.py --dataName daleN100_9fbe7f -m 0 -p a b d e
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
from pprint import pprint
import numpy as np
from PlotterDaleMatrix import Plotter
from toolbox.Util_NumpyIOv2 import read_data_npz
import argparse

#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser(description="Visualize simulated Dale Poisson network data")
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a b', nargs='+',help="abc-string listing shown plots: a=Dale_matrix_eigen_locations, b=histo_weights_rates, c=rates_study, d=pseudospectra")
    parser.add_argument("--plotFormat", choices=("png", "pdf"), default="png", help="Output format for saved plots")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--basePath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for input data")
    parser.add_argument("--dataName",  default='daleN150_448b86',help='simulated Dale network base name')
    #parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to basePath)")
    parser.add_argument('-m', '--idxState', type=int, default=0, help="Index into state list, selects which state to plot")
    
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
        print("\nSimulation Truth Metadata:");        pprint(trueMD)
    
    # Load spike data (generated spike counts and rates)
    spikesFF = os.path.join(args.inpPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb>0)
    if args.verb>1:
        print("\nSpike Data Metadata:");        pprint(spikeMD)
    
    # Merge metadata for plotting
    trueMD['short_name'] = args.dataName
    
    # Select B-offset slice
    ib = args.idxState
    boffsets = trueMD['dale_conf']['Boffsets']
    assert ib < len(boffsets), f"idxState={ib} out of range, only {len(boffsets)} states available"
    B_offset = boffsets[ib]
    R_sel = trueMD['dale_conf']['spectral_radius']
    print(f"\nSelected B offset [{ib}]: offset={B_offset:.3f}  (out of {boffsets})")
    print(f"Spectral radius: R={R_sel:.3f}")
    
    # Slice stacked arrays along axis 0 for the selected B offset
    trueD_r = { 'A_true': trueD['A_true'], 'B_true': trueD['B_true'][ib], 'E_true': trueD['E_true'] }
    for key in ('node_positions', 'node_is_inhibitory', 'node_distance_matrix'):
        if key in trueD:
            trueD_r[key] = trueD[key]
    spikeD_r = { k: spikeD[k][ib] for k in spikeD }
    trueMD['sel_spect_radius'] = R_sel
    trueMD['sel_Boffset'] = B_offset
    trueMD['sel_state'] = ib
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=args.dataName+'_view%d'%ib
    plot=Plotter(args)
    
    if 'a' in args.showPlots:
        plot.Dale_matrix_and_eigen(trueD_r['A_true'], trueMD, trueD_r, spikeD_r, figId=1)
    
    if 'b' in args.showPlots:
        plot.histo_weights_rates(trueD_r,spikeD_r,trueMD,figId=2)
 
    if 'c' in args.showPlots:
        plot.rates_study(trueD_r,spikeD_r,trueMD,figId=3)

    if 'd' in args.showPlots:
        plot.Dale_matrix_pseudospectra(trueD_r['A_true'],trueMD,trueD_r,figId=4)
 
    plot.display_all()
    print('M:done - view_dalePoisson completed successfully!')
