#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''

HD5 arrays contain input and output
Use sampler and manual transpiler
Dependence:  qiskit 1.2


Use case: XXX
./submit_ibmq_job.py -E  --numQubits 3 3 --numSample 15 --numShot 8000  --backend   ibm_brussels  


'''
import sys,os,hashlib
import numpy as np
import pickle
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

import argparse
#...!...!..................
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--inpPath",default='/global/cfs/cdirs/m2043/causal_inference/DIV13',help="raw input data")
    
    parser.add_argument("--sessionName",  default='HET_80k_1',help='raw data session name')
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument("--outName",  default=None,help='(optional) output file name')
 
    # .... activity speciffic speciffic, 
    parser.add_argument('--tau_decay_ms', default=[1.1, 10.],  nargs=2, type=float, help='Exponential decay constant and tail length')
    parser.add_argument('--time_rebin', default=1, type=int, help='rebin of raw time axis')
    parser.add_argument('-T','--maxTime', default=1.2, type=float, help='cut-off of time for raw data')
    parser.add_argument('--num_feature', default=10, type=int, help='num of from full dataset')

    args = parser.parse_args()
    args.inpPath='/dataVault2025/causalNet_tmp/'  # on laptop
    args.dataPath=os.path.join(args.basePath,'input')
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.dataPath)
    
    return args

#...!...!....................
def buildPayloadMeta(args):
    pd={}  # payload
    pd['raw_input_path']=args.inpPath
    pd['session_name']=args.sessionName
    pd['tau_decay']=[ x/1000. for x in args.tau_decay_ms]
    pd['max_time']=args.maxTime
    md={ 'payload':pd}
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash']=myHN
    if args.outName==None:
        md['short_name']='%s-%s'%(args.sessionName,md['hash'])
    else:
        md['short_name']=args.outName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md

#...!...!....................
def read_spike_dict(md,args):
    inpF=os.path.join(args.inpPath,args.sessionName,'spike_dict.pkl')
    print('inpF:',inpF)
    assert os.path.exists(inpF)
    # Load the dictionary from the .pkl file
    with open(inpF, "rb") as f:
        spike_dict = pickle.load(f)

    raw_sampling_freq=10000  # Hz
    pmd=md['payload']
    pmd['sampling_freq'] =raw_sampling_freq/args.time_rebin
    
    #....  select clip time bin
    clipTbin=int(pmd['max_time'] * pmd['sampling_freq'])
    pmd['num_tume_bin']=clipTbin
    pprint(pmd)
    
    # neuron ID  MEA chip
    meaIdL=np.array(sorted(spike_dict))

    # ... down select neurons
    if len(meaIdL) > args.num_feature: meaIdL=meaIdL[:args.num_feature]
    print('RSD: meaID list:',meaIdL)
    pmd['num_feature']=len(meaIdL)
    pmd['feature_id']=meaIdL
    
    spikeD={}
    for k in meaIdL:
        rec=np.array(spike_dict[k])/args.time_rebin
        rec2=rec[rec<clipTbin].astype(int)
        print('meaId:',k,len(rec),len(rec2))
        spikeD[k]=rec2
    print(rec2)
    
    return  spikeD

#...!...!....................
def build_decay_data(bSpikeD,md):
    pmd=md['payload']
    nfeat=pmd['num_feature']
    ntime=pmd['num_tume_bin']
    actA=np.zeros((nfeat,ntime),dtype=np.float16)
    spikeA=np.zeros((nfeat,ntime),dtype=np.bool_)
    for k in range(nfeat):
        print('k',k)
        fid=pmd['feature_id'][k]
        add_spike_decay(bSpikeD[fid],pmd['tau_decay'],pmd['sampling_freq'],actA[k])
        spikeA[k][bSpikeD[fid]]=True  # unpack spikes
        
    timeV = np.linspace(0, pmd['max_time'],  ntime)
    print('ttt',timeV[:5], timeV[-5:])
    print('qqq',bSpikeD[fid].shape, bSpikeD[fid].dtype)
    bigD={'feature':actA,'time':timeV,'spike':spikeA}
    return bigD
    
#...!...!....................
def add_spike_decay(bSpikeL,tauV,sampling_rate,dataV):
    """
    Adds exponential decay to each binary spike.
    - bSpikeL: list of time bins with spikes
    - sampling_rate: Sampling rate in Hz
    - tau_decay: Decay constant in seconds
    - tail_len
    """
    #y_pred = np.zeros_like(binary_data,dtype=np.float64)
    tau_decay,tail_len=tauV
    decay_samples = int(tail_len * sampling_rate)
    assert decay_samples>1  # decay is just a spike
    
    # Create an exponential decay curve
    decay_curve = np.exp(-np.arange(decay_samples) / (tau_decay * sampling_rate))
       
    # Apply exponential decay to each spike
    for i in bSpikeL:
        end = min(i + decay_samples, dataV.shape[0])
        #print(i,end)
        dataV[i:end] += decay_curve[:end-i]
    


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    args=commandline_parser()
    np.set_printoptions(precision=5)
    expMD=buildPayloadMeta(args)
   
    pprint(expMD)
    #=construct_random_inputs(expMD,args)

    # read raw data
    binSpikeD=read_spike_dict(expMD,args)
    expD=build_decay_data(binSpikeD,expMD)
    
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.dataPath,expMD['short_name']+'.act.h5')
    write4_data_hdf5(expD,outF,expMD)
    print('   ./plot_features.py  --inpName   %s   \n'%(expMD['short_name'] ))
    print('   ./fit_uoiVar.py  --inpName   %s   \n'%(expMD['short_name'] ))
    pprint(expMD)


    
