#!/usr/bin/env python3
"""
Visualization tool for biological experiment input features and data quality.

This script provides comprehensive visualization and analysis of experimental
neural data, focusing on data quality assessment and feature exploration.
Main functionality includes:
- Time series visualization of neural recordings with customizable time ranges
- Data quality metrics and statistical summaries
- Cluster detection and activity pattern analysis
- Interactive plotting with configurable display options

Used primarily for exploratory data analysis of biological neural recordings
before further processing and connectivity analysis.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os

from time import time
from pprint import pprint
import numpy as np
from PlotterBioExp import Plotter
from toolbox.Util_NumpyIO import read_data_npz
from UtilBioExp import detect_spike_bursts
import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")

    parser.add_argument("--dataPath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for any further data processing")

    parser.add_argument('-T','--time_range' , default=[0., 60],  nargs=2,   type=float, help='display data time range in seconds')
    parser.add_argument('--burst_freq_thres' , default=5.,    type=float, help='tags high freq channels for burts detection')
    parser.add_argument('--burst_chan_thres' , default=30,    type=int, help='burst flag when instant high-rate neuron count exceeds this')
  
    parser.add_argument("--dataName",  default='HET_80k_1-fc62ef',help='preprocessed  session name')

    parser.add_argument('-R','--time_rebin2', default=50, type=int, help='rebin current time axis')
   
    args = parser.parse_args()
    # make arguments  more flexible
    args.outPath='out/'
    args.showPlots=''.join(args.showPlots)
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)

    spikesFF = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF)
    if args.verb>1: pprint(spikeMD)
    rebD = detect_spike_bursts(
        spikeD, spikeMD, args.time_rebin2,
        args.burst_freq_thres, args.burst_chan_thres,
    )

    # ... for plotting
    bioFF=os.path.join(args.dataPath, f"{args.dataName}.bioExp.npz")  
    print('ttt',bioFF)
    bioD, bioMD = read_data_npz(bioFF)

    #--------------------------------
    # ....  plotting ........
    args.prjName = spikeMD['provenance']['experiment_name']
    spikeMD['plot']={}    
    spikeMD['plot']['time_rangeLR']=np.array(args.time_range)
    spikeMD.update(**bioMD)
    
    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.freq_histo(spikeD,spikeMD,figId=1)
    if 'b' in args.showPlots:
        plot.freq_vs_time(rebD,spikeMD,figId=2)

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
