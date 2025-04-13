#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

'''

import os
import pickle
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

from time import time
from pprint import pprint
import numpy as np
from Plotter_Dale_LDS import Plotter
from time import time
import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")         
    parser.add_argument("--basePath",default='dataDale',help="head dir for set of experimentst")
    parser.add_argument("--simName", default='daleM60r5', help="[.h5] Dale Matrix file name")
         
    args = parser.parse_args()
    # make arguments  more flexible
    args.inpPath=os.path.join(args.basePath,'simu')
    args.outPath=os.path.join(args.basePath,'out')
    args.showPlots=''.join(args.showPlots)
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def postproc_spikes(bigD,md):
    sm=md['simu']
    dt=sm['time_step']
    countRaw=bigD['Xcount']
    nShot=countRaw.shape[0]
    assert nShot==sm['num_trials']
    countSum=np.sum(countRaw,axis=0)
    fac=dt*nShot
    print('countSum:',countSum.shape,' dt:%.2f  nShot=%d  fac=%.3f'%(dt,nShot,fac))
    rate=countSum/fac
    rateEr=np.sqrt(countSum)/fac
    bigD['Xrate']=rate
    bigD['XreateEr']=rateEr

    # evoked energy  
    ene=np.sum(rate**2,axis=1)

    bigD['raw_energy']=ene  # tmp: missing normalization factor
    return
    nTime=50
    t0=0
    print('evoked Ene (a.u.):',ene[t0:t0+nTime])
    for i in range(1):
        iNeur=i*10
        print('\n iNeur=%d '%(iNeur))
        print('countSum:',countSum[t0:t0+nTime,iNeur])
        print('rate(Hz):',rate[t0:t0+nTime,iNeur])
        print('rateEr(Hz):',rateEr[t0:t0+nTime,iNeur])
    ww



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)

    inpF=os.path.join(args.inpPath,args.simName+'.netActS.h5')
    bigD,MD=read4_data_hdf5(inpF)
    pprint(MD)

    postproc_spikes(bigD,MD)

    
    #--------------------------------
    # ....  plotting ........
    args.prjName=MD['short_name']
    #['plot']={}
    #if args.time_range!=None: expMD['plot']['time_rangeLR']=args.time_range

    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        #plot.rate_sample(bigD,MD,nidxL=[1,5,13,117,126,198],figId=1)  # 200-neurons
        plot.rate_sample(bigD,MD,nidxL=[1,5,13,37,46,54],figId=1)  # 60-neurons

    if 'b' in args.showPlots:
        plot.evoked_energy(bigD,MD,figId=1)


    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
