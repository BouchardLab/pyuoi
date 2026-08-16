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
from toolbox.Util_NumpyIOv2 import read_data_npz
import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser(description="Visualize spike-train data from simulated Dale network outputs")
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")

    parser.add_argument("--basePath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for input data")
    parser.add_argument("--dataName",  default=None,help='simulated Dale network base name')
  
    parser.add_argument('-m', '--idxState', type=int, default=0, help="Index into state list; if idxState<0 read spikes from spikesData/")

    parser.add_argument('-T','--time_range_sec' , default=[0., 50],  nargs=2,   type=float, help='display data time range in seconds')
    parser.add_argument('-r','--time_rebin2', default=5, type=int, help='rebin current time axis')
   
    args = parser.parse_args()
    # make arguments more flexible
    if args.idxState >= 0:
        args.inpPath = os.path.join(args.basePath, 'truthDale')
    else:
        args.inpPath = os.path.join(args.basePath, 'spikesData')
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
def XXXselect_radius_slice(spikeD, data_path, data_name, idxState, verb=1):
    """Select one spectral-radius slice from stacked spikes arrays if present."""
    spikes = spikeD['spikes']
    if spikes.ndim == 2:
        return spikeD, None
    assert spikes.ndim == 3, f"Expected spikes ndim 2 or 3, got shape={spikes.shape}"

    nR = spikes.shape[0]
    assert 0 <= idxState < nR, f"idxState={idxState} out of range for spikes with nR={nR}"

    truthFF = os.path.join(data_path, f"{data_name}.simTruth.npz")
    assert os.path.exists(truthFF), f"missing simTruth file: {truthFF}"
    _, trueMD = read_data_npz(truthFF, verb=verb>0)
    spect_radii = trueMD['dale_conf']['spect_radius']
    assert idxState < len(spect_radii), f"idxState={idxState} out of range, only {len(spect_radii)} states available"
    R_sel = spect_radii[idxState]
    print(f"\nSelected spectral radius [{idxState}]: R={R_sel:.3f}  (out of {spect_radii})")

    spikeD_r = {}
    for key, arr in spikeD.items():
        if isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] == nR:
            spikeD_r[key] = arr[idxState]
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
   
    S_oracle = None
    if args.idxState >=0:  # per state information
          # Load simulation truth data (Dale matrices, biases, etc.)
        truthFF = os.path.join(args.inpPath, f"{args.dataName}.simTruth.npz")
        trueD, trueMD = read_data_npz(truthFF, verb=args.verb>0)
        if args.verb>1: 
            print("\nSimulation Truth Metadata:");        pprint(trueMD)
            
        dataYield = spikeD['spikes']
        Mstate,Nt, Nn = dataYield.shape
        m=args.idxState
        assert m<Mstate
        spikeD['spikes']=dataYield[m]
        spikeD['single_rates']=spikeD['single_rates'][m]
        spikeMD['sel_spect_radius'] = trueMD['dale_conf']['spectral_radius']
        S_true = None
        
    if args.idxState <0:  #  multi-state simulations
        prismFF = os.path.join(args.inpPath, f"{args.dataName}.prismTruth.npz")
        assert os.path.exists(prismFF), f"missing prismTruth file: {prismFF}"
        prismD, prismMD = read_data_npz(prismFF, verb=args.verb>0)
        S_true = prismD['S_true']
        S_oracle = prismD['S_oracle']
        if args.verb > 0:  print('loaded prismTruth S_true:', S_true.shape);
        if args.verb > 1:  pprint(prismMD)
        spikeMD['sel_spect_radius'] =-77
        oraE = prismMD['oracle_eval']
        spikeMD['oracle_score'] = oraE['avr_score']
        print(f"gen, oracle avr score {oraE['avr_score']:.3f}, {args.dataName}")
        print(f"  {'state':>5s}  {'score':>5s}")
        print(f"  {'-----':>5s}  {'-----':>5s}")
        for m, sc in enumerate(oraE['score_per_state']):
            score_text = "  n/a" if sc is None else f"{sc:5.3f}"
            print(f"  {m:5d}  {score_text}")

    #--------------------------------
    # ....  plotting ........
    spikeMD['short_name']=f"{args.dataName}_state{args.idxState}"
    spikeMD['sel_state'] = int(args.idxState)
    args.prjName=spikeMD['short_name']
    spikeMD['plot']={}    
    spikeMD['plot']['time_rangeLR']=np.array(args.time_range_sec)
    
    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.freq_histo(spikeD,spikeMD,figId=1)
    if 'b' in args.showPlots:
        rebD=rebin_spike_rates(spikeD['spikes'], spikeMD, args.time_rebin2)
        if S_oracle is not None:
            rebD['S_oracle'] = S_oracle
        plot.freq_vs_time(rebD,spikeMD,figId=2, S_true=S_true, S_oracle=S_oracle)

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
