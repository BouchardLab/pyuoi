#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

'''

import os

from time import time
from pprint import pprint
import numpy as np
from PlotterBioExp import Plotter
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from UtilBioExp import create_clusters_mask
import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")

    parser.add_argument("--dataPath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for any further data processing")

    parser.add_argument('-T','--time_range' , default=[0., 600],  nargs=2,   type=float, help='display data time range in seconds')
    parser.add_argument('--burst_freq_thres' , default=40.,    type=float, help='tags high freq channels for burts detection')
    parser.add_argument('--burst_chan_thres' , default=30,    type=int, help='final thres eliminating time bins')
  
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


#...!...!....................
def detect_spike_bursts(spikeD, md):
    rateThr2=args.burst_freq_thres
    spikeYield = spikeD['spikes'] #, spikeD['single_rates']
    time_step=md['time_step_sec']
    tReb2=args.time_rebin2
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
    mask2D=rate2D>rateThr2
    highChan=np.sum(mask2D,axis=1)

    #.... compute running sume over K bins
    K = 5  # Number of bins for the running sum
    mCnt=args.burst_chan_thres # minimal number of highRate neurons to flag the cluster in time
    # Create a kernel for the running sum
    kernel = np.ones(K)/K

    # Compute the running sum using convolution
    XS = np.convolve(highChan, kernel, mode='valid')
    XS = np.pad(XS, (K-1, 0), mode='constant', constant_values=0)
    XM=create_clusters_mask(XS,th=mCnt)
    usableFrac=1-np.sum(XM)/XM.shape[0]
    
    rebD={'time_step2':time_step2,'rate_thres2':rateThr2,'smooth_kernel':K,'high_cnt_thres':mCnt,'usable_time_fract':usableFrac}
    rebD['rate2D']=rate2D
    rebD['mask2D']=mask2D
    rebD['highChanCnt']=highChan
    rebD['highChanSmooth']=XS
    rebD['highChanMask']=XM
    rebD['rate1D']=np.sum(rate2D,axis=1)
    ntime//=tReb2
    rebD['timeV']= np.linspace(0, (ntime - 1) * time_step2, ntime)

    print('usable time frac:%.3f  nchan=%d  thr=%.1f Hz'%(rebD['usable_time_fract'],nchan, rateThr2))
    timeMask=np.repeat(XM, tReb2)
    return rebD,timeMask

#...!...!....................
def filter_bursts(spikeD, md,timeMask):
    pprint(md)
    spikeD['spikes'][:len(timeMask)][timeMask]=0
    
  
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
    
    rebD,timeMask=detect_spike_bursts(spikeD, spikeMD)

     #...... WRITE   OUTPUT .........
    maskFF=spikesFF.replace('spikes','timeMask')
    write_data_npz({'time_mask':timeMask}, maskFF, metaD=None)

    filter_bursts(spikeD, spikeMD,timeMask)
          
    #--------------------------------
    # ....  plotting ........
    args.prjName=spikeMD['short_name']
    spikeMD['plot']={}
    
    spikeMD['plot']['time_rangeLR']=np.array(args.time_range)

    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.freq_histo(spikeD,spikeMD,figId=1)
    if 'b' in args.showPlots:
        plot.freq_vs_time(rebD,spikeMD,figId=2)

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
