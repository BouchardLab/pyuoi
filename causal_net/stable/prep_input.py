#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''

HD5 arrays contain input and output
Use sampler and manual transpiler
Dependence:  qiskit 1.2


Use case:

basePath=/global/cfs/cdirs/m2043/causal_inference/DIV13
ses=HET_80k_1 ; ses2=${ses}_samp1kHz
./prep_input.py --sessionName $ses --outName $ses2  --basePath $basePath
./plot_features.py   --basePath $basePath --inpName   $ses2 -p  d -Y


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
    parser.add_argument("--inpPath",default='/m2043/DIV13/',help="raw input data")
    
    parser.add_argument("--sessionName",  default='HET_80k_1',help='raw data session name')
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument("--outName",  default=None,help='(optional) output file name')
 
    # .... activity speciffic speciffic, 
    parser.add_argument('--tau_decay_ms', default=[1., 10.],  nargs=2, type=float, help='Exponential decay constant and tail length')
    parser.add_argument('--time_rebin', default=10, type=int, help='rebin of raw time axis')
    parser.add_argument('--num_feature', default=None, type=int, help='num of input features from full dataset')

    args = parser.parse_args()
    #args.inpPath='/dataVault2025/causalNet_tmp'  # on laptop
    args.inpPath='/global/cfs/cdirs/m2043/causal_inference/DIV13'  # bare PM
    args.dataPath=os.path.join(args.basePath,'features')
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
    md={ 'payload':pd}
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:7]
    md['hash']=myHN
    if args.outName==None:
        md['short_name']='%s-%s'%(args.sessionName,md['hash'])
    else:
        md['short_name']=args.outName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md

#...!...!....................
def qa_neuron(fid,spikeT): #???
    print(fid,spikeT.shape)
    print('qa fid=%d nspike=%d'%(fid,spikeT.shape[0]))
    delT=spikeT[1:] - spikeT[:-1]
    mind=np.min(delT)
    #imin=delT.index(mind)
    imin=int(np.where(delT == mind)[0] ) # first occurence
    print('mind:',mind,'imin=',imin)
    for i in range(imin-2,imin+3):
        print(i,spikeT[i]) #,spikeT[i+1]-spikeT[i])
    rrr

#...!...!....................
def read_spike_dict(md,args):
    pmd=md['payload']
    inpF=os.path.join(args.inpPath,args.sessionName,'spike_dict.pkl')
    print('inpF:',inpF)
    assert os.path.exists(inpF)
    # Load the dictionary from the .pkl file
    with open(inpF, "rb") as f:
        spike_dict = pickle.load(f)

    raw_sampling_freq=10000  # Hz
    assert raw_sampling_freq%args.time_rebin==0 
    pmd['sampling_freq'] =raw_sampling_freq/args.time_rebin   
    
    # neuron ID  MEA chip
    meaIdL=np.array(sorted(spike_dict))  # here order of feature_id is settled
    maxFeat=len(meaIdL)
    # ... down select neurons
    if  args.num_feature!=None:  meaIdL=meaIdL[:args.num_feature]
    if args.verb>1: print('RSD: meaID list:',meaIdL)
    pmd['num_feature']=len(meaIdL)
    pmd['feature_id']=meaIdL
    
    spikeD={}
    spikeCntL=np.zeros(pmd['num_feature'],dtype=int)  # num spikes per  neuron
    maxTbin=0
    dead_idL=[]
    j=0
    for k in meaIdL:
        rec=np.array(spike_dict[k])/args.time_rebin        
        spikeD[k]=rec.astype(int) # time-bin may repeat 
        spikeCntL[j]=len(rec)
        j+=1
        if len(rec)==0:
            dead_idL.append(int(k))
            continue        
        mxTb=np.max(rec)
        if maxTbin< mxTb: maxTbin=mxTb
        #... check for smalest dist
        #qa_neuron(k,spikeD[k])
        
    pmd['num_time_bin']=int(maxTbin)+1
    pmd['max_time']=pmd['num_time_bin']/pmd['sampling_freq']
    pmd['dead_id']=dead_idL
    return  spikeD,spikeCntL

#...!...!....................
def build_decay_data(bSpikeD,md): 
    pmd=md['payload']
    nfeat=pmd['num_feature']
    ntime=pmd['num_time_bin']
    actA=np.zeros((nfeat,ntime),dtype=np.float16)
    spikeA=np.zeros((nfeat,ntime),dtype=np.bool_)
    for k in range(nfeat):
        #print('k',k)
        fid=pmd['feature_id'][k]
        add_spike_decay(bSpikeD[fid],pmd['tau_decay'],pmd['sampling_freq'],actA[k])
        spikeA[k][bSpikeD[fid]]=True  # unpack spikes
        
    timeV = np.linspace(0, pmd['max_time'],  ntime)
    #print('ttt',timeV[:5], timeV[-5:])
    #print('qqq',bSpikeD[fid].shape, bSpikeD[fid].dtype)
    bigD={'feature':actA,'time':timeV,'spike':spikeA}
    return bigD
    
#...!...!....................
def add_spike_decay(bSpikeV,tauV,sampling_rate,dataV):
    """
    Adds exponential decay to each binary spike.
    - bSpikeL: list of time bins with spikes
    - sampling_rate: Sampling rate in Hz
    - tau_decay: Decay constant in seconds
    - tail_len
    """
   
    tau_decay,tail_len=tauV
    decay_samples = int(tail_len * sampling_rate)
    assert decay_samples>1  # decay is just a spike
    
    # Create an exponential decay curve
    decay_curve = np.exp(-np.arange(decay_samples) / (tau_decay * sampling_rate))
       
    # Apply exponential decay to each spike
    for i in bSpikeV:
        end = min(i + decay_samples, dataV.shape[0])
        #print(i,end)
        dataV[i:end] += decay_curve[:end-i]
    
#...!...!....................
def mon_spike_freq(bSpikeD,md,twindow_sec=60.): 
    pmd=md['payload']
    pmd['qa_twindow_sec']=twindow_sec
    nfeat=pmd['num_feature']
    ntime=pmd['num_time_bin']
    twindow_bin=int( pmd['sampling_freq'] * twindow_sec)
    num_window=1+ntime//twindow_bin
    freqA=np.zeros((nfeat,num_window),dtype=np.float32)
    for k in range(nfeat):        
        fid=pmd['feature_id'][k]
        bSpikeV=bSpikeD[fid]//twindow_bin
        countV = np.bincount(bSpikeV)
        mxb=countV.shape[0]
        
        #print('ss',bSpikeV.shape,num_window,mxb);
        #print(freqV[:30])
        freqA[k][:mxb]=countV/twindow_sec
       
    return freqA



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    args=commandline_parser()
    np.set_printoptions(precision=5)
    expMD=buildPayloadMeta(args)
   
    # read raw data
    binSpikeD,spikeCntL=read_spike_dict(expMD,args)
    #pprint(expMD)
    #pprint(spikeCntL)
    expD=build_decay_data(binSpikeD,expMD)
    #... QA
    expD['spike_freq']=mon_spike_freq(binSpikeD,expMD,twindow_sec=5.)   

    # it is too long , displays badly, move it to big data
    for xx in [ 'feature_id', 'dead_id']:
        expD[xx]=np.array( expMD['payload'].pop(xx),dtype=int)
    expD['avr_spike_freq']=spikeCntL/expMD['payload']['max_time']

     
    pprint(expMD)
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.dataPath,expMD['short_name']+'.act.h5')
    write4_data_hdf5(expD,outF,expMD)
    print('   ./plot_features.py  --inpName   %s  -p a b  -Y '%(expMD['short_name'] ))
    print('   ./fit_uoiVar.py  --inpName   %s   \n'%(expMD['short_name'] ))
   


    
