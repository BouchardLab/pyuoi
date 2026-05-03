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
from PlotterSpikesTrain import Plotter
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
  
    
    parser.add_argument('-T','--time_range_sec' , default=[0., 50],  nargs=2,   type=float, help='display data time range in seconds')
    parser.add_argument('-r','--time_rebin2', default=10, type=int, help='rebin current time axis')
   
    args = parser.parse_args()
    # make arguments more flexible
    args.inpPath = os.path.join(args.basePath, 'truthDale')
    
    #  args.inpPath = os.path.join(args.basePath, 'spikesData')
    args.outPath = os.path.join(args.basePath,'plots')
    args.showPlots=''.join(args.showPlots)
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    if args.time_range_sec!=None: assert args.time_range_sec[0] < args.time_range_sec[1] 
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
    pop_spike_count = np.sum(spikeYieldR, axis=1)
    pop_rate_hz = pop_spike_count / time_step2

    rebD={'time_step2':time_step2}
    rebD['rate2D']=rate2D
    rebD['pop_spike_count'] = pop_spike_count
    rebD['pop_rate_hz'] = pop_rate_hz
    ntime//=tReb2
    rebD['timeV']= np.linspace(0, (ntime - 1) * time_step2, ntime)

    print('rebinned rates: nchan=%d  dt=%.3f sec  rebin=%d' % (nchan, time_step2, tReb2))
    return rebD


#...!...!....................
def rebin_network_state(state2D, md, tReb2, state_name='state'):
    """Rebin a continuous network state (time x neurons) by averaging over time."""
    state2D = np.asarray(state2D)
    assert state2D.ndim == 2, f"Expected 2D {state_name}, got shape={state2D.shape}"
    assert tReb2 < 101  # keep consistent with spike-rate rebinning guard

    ntime, nchan = state2D.shape
    if ntime % tReb2 != 0:
        ntime_c = ntime - (ntime % tReb2)
        state2D = state2D[:ntime_c]
        ntime = ntime_c

    state2DR = np.mean(state2D.reshape(-1, tReb2, nchan), axis=1)
    time_step = md['time_step_sec']
    time_step2 = time_step * tReb2
    ntime2 = state2DR.shape[0]

    rebD = {
        'time_step2': time_step2,
        'state2D': state2DR,
        'timeV': np.linspace(0, (ntime2 - 1) * time_step2, ntime2),
        'state_name': state_name,
    }

    print('rebinned %s: nchan=%d  dt=%.3f sec  rebin=%d' % (state_name, nchan, time_step2, tReb2))
    return rebD


  
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
   
    if 1:  # per state information
          # Load simulation truth data (Dale matrices, biases, etc.)
        truthFF = os.path.join(args.inpPath, f"{args.dataName}.simTruth.npz")
        trueD, trueMD = read_data_npz(truthFF, verb=args.verb>0)
        if args.verb>1: 
            print("\nSimulation Truth Metadata:");        pprint(trueMD)
            
        dataYield = np.asarray(spikeD["spikes"])
        if dataYield.ndim != 2:
            raise ValueError(
                "spikes must have shape (T, N) from gen_daleMatrices4; got %s. Multi-state (M, T, N) is not supported."
                % (dataYield.shape,)
            )
        sr = np.asarray(spikeD["single_rates"])
        if sr.ndim != 1:
            raise ValueError("single_rates must be (N,); got shape %s" % (sr.shape,))
        if sr.shape[0] != dataYield.shape[1]:
            raise ValueError("single_rates length must match spikes.shape[1]")
        spikeD["spikes"] = dataYield
        spikeD["single_rates"] = sr
        spikeMD["sel_spect_radius"] = trueMD["dale_conf"]["spectral_radius"]
        
 
    #--------------------------------
    # ....  plotting ........
    spikeMD["short_name"] = args.dataName
    args.prjName = spikeMD["short_name"]
    spikeMD['plot']={}    
    spikeMD['plot']['time_rangeLR']=np.array(args.time_range_sec)
    
    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.freq_histo(spikeD,spikeMD,figId=1)
    if 'b' in args.showPlots:
        rebD=rebin_spike_rates(spikeD['spikes'], spikeMD, args.time_rebin2)
        plot.freq_vs_time(rebD,spikeMD,figId=2)
    if 'c' in args.showPlots:
        rebX = rebin_network_state(trueD['x_true'], spikeMD, args.time_rebin2, state_name='x')
        plot.state_x_vs_time(rebX, spikeMD, figId=3)

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
