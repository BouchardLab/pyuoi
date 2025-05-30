#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

'''

import os
import pickle
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_CausalNet import rebin_axis0_average
from Plotter_Dale_LDS import Plotter

from time import time
from pprint import pprint
import numpy as np

from time import time
import argparse

#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    parser.add_argument('--time_range' , default=[0.3, 1.],  nargs=2,   type=float, help='fit data time range')
     
    parser.add_argument("--basePath",default='dataDale',help="head dir for set of experimentst")
    parser.add_argument("--simName", default='daleM100apr30-e7e3be2', help="[.h5]  simulated netActivation")
    parser.add_argument("--time_rebin", type=int, default=1, help="num time steps to be averaged")
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")         
         
    args = parser.parse_args()
    # make arguments  more flexible
    args.inpPath=os.path.join(args.basePath,'gen_dale')
    args.outPath=os.path.join(args.basePath,'postproc')
    args.showPlots=''.join(args.showPlots)
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
   
    return args


#...!...!....................
def postproc_netActivity(bigD,md):
    sim=md['simu']
    dt=sim['time_step']
    pom={}
    md['postproc']=pom
    
    timeV=bigD['evol_time']
    
    #.... clip data in time
    tL,tR=[int(x/dt) for x in args.time_range ]
    print('FUV tbinLR:',tL,tR)
    assert tR <= timeV.shape[0]
    timeV=timeV[tL:tR]
    stateV=bigD['evol_state'][tL:tR]
    pom['time_range']=[args.time_range[0], args.time_range[1]]
    pom['time_rebin']=args.time_rebin

    if args.time_rebin>1:  # averag data over time        
        pom['time_step']=args.time_rebin*dt
        timeV= rebin_axis0_average(timeV, args.time_rebin)
        stateV= rebin_axis0_average(stateV, args.time_rebin)    
    
    rateV=np.exp(stateV)

    # overwrite data
    bigD['evol_time']=timeV
    bigD['evol_state']=stateV
    bigD['evol_rate']=rateV
    
    # evoked energy  
    ene=np.sum(rateV**2,axis=1)
    bigD['raw_energy']=ene  # tmp: missing normalization factor
    return


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)

    inpF=os.path.join(args.inpPath,args.simName+'.simNet.h5')
    bigD,MD=read4_data_hdf5(inpF)
    pprint(MD)

    postproc_netActivity(bigD,MD)
    numNeur=MD['dale_truth']['num_any_neur']
    
    nidxL=[1,5,13,37,46,54] # use this if you want fixed neurons instead of random
    nidxL=np.sort(np.random.choice(numNeur, size=6, replace=False))
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=MD['short_name']
    MD['plot']={}
     
    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.rate_sample(bigD,MD,nidxL=nidxL,figId=1,obsN='rate') 
    if 'b' in args.showPlots:
        plot.rate_sample(bigD,MD,nidxL=nidxL,figId=2,obsN='state')
        
    if 'c' in args.showPlots:
        plot.rate_correl(bigD,MD,nidxL=nidxL,obsN='state',figId=3) 

    if 'e' in args.showPlots:
        plot.evoked_energy(bigD,MD,figId=1)


    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
