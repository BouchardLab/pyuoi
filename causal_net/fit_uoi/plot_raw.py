#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

'''

import os
import pickle
#from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from time import time
from pprint import pprint
import numpy as np
from PlotterRaw_QA import Plotter

import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")         
    parser.add_argument("--basePath",default='out',help="head dir for set of experimentst")
    parser.add_argument('--time_range' , default=[0., 5.0],  nargs=2,   type=float, help='fit data time range')
    parser.add_argument('-f','--num_feature', default=10, type=int, help='num of features from full dataset, 0 is all')
    parser.add_argument("--sessionName",  default='HET_80k_1',help='raw data session name')
    
    args = parser.parse_args()
    # make arguments  more flexible
    args.inpPath='/global/cfs/cdirs/m2043/causal_inference/DIV13'  # bare PM
    args.outPath='out/raw_qa'
    args.showPlots=''.join(args.showPlots)
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def read_spike_dict(args):
    inpF=os.path.join(args.inpPath,args.sessionName,'spike_dict.pkl')
    print('inpF:',inpF)
    assert os.path.exists(inpF)
    # Load the dictionary from the .pkl file
    with open(inpF, "rb") as f:
        spike_dict = pickle.load(f)

    args.raw_sampling_freq=10000  # Hz
    return spike_dict

#...!...!....................
def total_spike_count(bigD):
    spikeCntV=np.zeros(num_feature,dtype=int)
    spikeMinDT=np.zeros_like(spikeCntV)
    maxTbin=0
    dead_idL=[]
    for j,fid in enumerate(meaIdL):
        rec=np.array(spikeD[fid],dtype=int)
        spikeCntV[j]=len(rec)
        if len(rec)==0:
            dead_idL.append(int(fid))
            continue        
        mxTb=np.max(rec)
        if maxTbin< mxTb: maxTbin=mxTb
        #... check for smalest dist
        delV=rec[1:] - rec[:-1]
        dtm=np.min(delV)
        spikeMinDT[j]=dtm
        #print('j,fid,dtm,nspike',j,fid,dtm,spikeCntV[j])
    bigD['tot_spike_vs_fid']=spikeCntV
    bigD['min_spike_delT']=spikeMinDT
    
 
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)

    spikeD=read_spike_dict(args)
    print(type(spikeD))
    
    # neuron ID  MEA chip
    meaIdL=np.array(sorted(spikeD))  # here order of feat_id is settled
    maxFeat=len(meaIdL)
    # ... down select neurons
    if  args.num_feature>0:  meaIdL=meaIdL[:args.num_feature]
    #print('RSD: meaID list:',meaIdL)
    num_feature=len(meaIdL)

    expD={}
    expMD={'short_name':args.sessionName}
    # .... spike count
    total_spike_count(expD)
    
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
     #['plot']={}
    
    #if args.time_range!=None: expMD['plot']['time_rangeLR']=args.time_range

    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.spikes_general(expD,expMD,figId=1)
    if 'b' in args.showPlots:
        plot.input_features_dense(expD,expMD,figId=2)

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
