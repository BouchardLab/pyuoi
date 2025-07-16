#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

'''

import os
import pickle
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from Util_CausalNet import rebin_axis0_average
from PlotterDaleLDS import Plotter

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
    parser.add_argument('--time_range' , default=[0, 500],  nargs=2,   type=int, help=' data time range in bins')
     
    parser.add_argument("--basePath",default='dataDale',help="head dir for set of experimentst")
    parser.add_argument("--simName", default='daleM100apr30-e7e3be2', help="[.h5]  simulated netActivation")
    
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
    sim=md['dataset']
    assert 'simu' in sim['type']
    pom={}
    md['postproc']=pom

    nStep=sim['num_time_steps']
    timeV=np.linspace(0, nStep-1,nStep)
    
    #.... clip data in time
    tL,tR= args.time_range 
    #print('FUV tbinLR:',tL,tR)
    assert tR <= timeV.shape[0]
    timeV=timeV[tL:tR]
    stateV=bigD['simu_state'][:,tL:tR]
    spikeV=bigD['simu_spikes'][:,tL:tR]
    pom['time_range']=[args.time_range[0], args.time_range[1]]
    
    rateV=np.exp(stateV)

    # overwrite data
    bigD['evol_time']=timeV  # do I need it?
    bigD['simu_state']=stateV
    bigD['simu_spikes']=spikeV
    
    
    # evoked energy  
    ene=np.sum(rateV**2,axis=0)
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
        plot.rate_sample(bigD,MD,nidxL=nidxL,figId=1,obsN='state') 
    if 'b' in args.showPlots:
        plot.rate_sample(bigD,MD,nidxL=nidxL,figId=2,obsN='rate')
    if 'c' in args.showPlots:
        plot.rate_sample(bigD,MD,nidxL=nidxL,figId=3,obsN='spikes')
        
    if 'd' in args.showPlots:
        plot.evoked_energy(bigD,MD,figId=4)


    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
