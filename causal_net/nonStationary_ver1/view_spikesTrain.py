#!/usr/bin/env python3
"""
Visualization tool for biological experiment input features and data quality.

This script provides comprehensive visualization and analysis of experimental
neural data, focusing on data quality assessment and feature exploration.
Main functionality includes:
- Time series visualization of neural recordings with customizable time ranges
- Data quality metrics and statistical summaries
- Interactive plotting with configurable display options

Used primarily for exploratory data analysis of biological neural recordings
before further processing and connectivity analysis.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os

from pprint import pprint
import numpy as np
from PlotSpikesTrain import Plotter
from toolbox.Util_NumpyIO import read_data_npz
import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser(description="Visualize spike-train data from simulated Dale network outputs")
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")

    parser.add_argument("--basePath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for input data")
    parser.add_argument("--dataName",  default=None,help='simulated Dale network base name')
  
    parser.add_argument('-i', '--idxR', type=int, default=0, help="Index into spect_radius list; if idxR<0 read spikes from spikesData/")

    parser.add_argument('-T','--time_range' , default=[0., 600],  nargs=2,   type=float, help='display data time range in seconds')
    parser.add_argument('-r','--time_rebin2', default=50, type=int, help='rebin current time axis')
   
    args = parser.parse_args()
    # make arguments more flexible
    if args.idxR >= 0:
        args.inpPath = os.path.join(args.basePath, 'truthDale')
    else:
        args.inpPath = os.path.join(args.basePath, 'spikesData')
    args.outPath = os.path.join(args.basePath,'plots')
    args.showPlots=''.join(args.showPlots)
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    assert os.path.exists(args.basePath)
    assert os.path.exists(args.inpPath), f"missing inpPath: {args.inpPath}"
    return args


#...!...!....................
def rebin_spike_rates(spikeYield, md, tReb2):
    """Rebin 2D spike train (time x neurons) and return rate summaries over rebinned time."""
    assert spikeYield.ndim == 2, f"Expected 2D spike train, got shape={spikeYield.shape}"
    assert tReb2<101  # this would exceed 1 seconds

    #.... rebin time axis
    ntime,nchan=spikeYield.shape
    if ntime % tReb2 != 0:
            ntime_c = ntime - (ntime % tReb2)
            spikeYield = spikeYield[:ntime_c]  # Clip the data
            #print('ccl',ntime_c , ntime ,ntime % tReb2, tReb2)
    #...  sum yileds over tReb2 time bins
    spikeYieldR=    np.sum( spikeYield.reshape(-1, tReb2, nchan),axis=1)
    time_step=md['time_step_sec']
    time_step2=time_step*tReb2
    #print('rr1',time_step2,spikeYieldR.shape,spikeYield.shape)

    rate2D=spikeYieldR/time_step2

    rebD={'time_step2':time_step2}
    rebD['rate2D']=rate2D
    ntime//=tReb2
    rebD['timeV']= np.linspace(0, (ntime - 1) * time_step2, ntime)

    print('rebinned rates: nchan=%d  dt=%.3f sec  rebin=%d' % (nchan, time_step2, tReb2))
    return rebD

#...!...!....................
def select_radius_slice(spikeD, data_path, data_name, idxR, verb=1):
    """Select one spectral-radius slice from stacked spikes arrays if present."""
    spikes = spikeD['spikes']
    if spikes.ndim == 2:
        return spikeD, None
    assert spikes.ndim == 3, f"Expected spikes ndim 2 or 3, got shape={spikes.shape}"

    nR = spikes.shape[0]
    assert 0 <= idxR < nR, f"idxR={idxR} out of range for spikes with nR={nR}"

    truthFF = os.path.join(data_path, f"{data_name}.simTruth.npz")
    assert os.path.exists(truthFF), f"missing simTruth file: {truthFF}"
    _, trueMD = read_data_npz(truthFF, verb=verb>0)
    spect_radii = trueMD['dale_conf']['spect_radius']
    assert idxR < len(spect_radii), f"idxR={idxR} out of range, only {len(spect_radii)} radii available"
    R_sel = spect_radii[idxR]
    print(f"\nSelected spectral radius [{idxR}]: R={R_sel:.3f}  (out of {spect_radii})")

    spikeD_r = {}
    for key, arr in spikeD.items():
        if isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] == nR:
            spikeD_r[key] = arr[idxR]
        else:
            spikeD_r[key] = arr
    return spikeD_r, R_sel


  
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)

    spikesFF = os.path.join(args.inpPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb>0)
    if args.verb>1: pprint(spikeMD)

    S_true = None
    if args.idxR < 0:
        prismFF = os.path.join(args.inpPath, f"{args.dataName}.prismTruth.npz")
        assert os.path.exists(prismFF), f"missing prismTruth file: {prismFF}"
        prismD, prismMD = read_data_npz(prismFF, verb=args.verb>0)
        S_true = prismD['S_true']
        if args.verb > 0:
            print('loaded prismTruth S_true:', S_true.shape)
        if args.verb>1: pprint(prismMD)

    spikeD_r, R_sel = select_radius_slice(spikeD, args.inpPath, args.dataName, args.idxR, verb=args.verb)
    spikeMD['sel_spect_radius'] = R_sel
    
    #--------------------------------
    # ....  plotting ........
    spikeMD['short_name']
  

    args.prjName=f"{spikeMD['short_name']}_view{args.idxR}"
    spikeMD['plot']={}    
    spikeMD['plot']['time_rangeLR']=np.array(args.time_range)
    
    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.freq_histo(spikeD_r,spikeMD,figId=1)
    if 'b' in args.showPlots:
        rebD=rebin_spike_rates(spikeD_r['spikes'], spikeMD, args.time_rebin2)
        plot.freq_vs_time(rebD,spikeMD,figId=2, S_true=S_true)

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
