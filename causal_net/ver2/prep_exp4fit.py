#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
prepares input for UoI fit experimental data

HD5 arrays contain input and output

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
    parser.add_argument("--inpPath",default='/global/cfs/cdirs/m2043/causal_inference/DIV13',help="raw input data")
    
    parser.add_argument("--sessionName",  default='HET_80k_1',help='raw data session name')
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")    
    parser.add_argument("--outName",  default=None,help='(optional) output file name')
 
   
    args = parser.parse_args()
    args.time_rebin=10 # 'rebin of raw time axis'
    args.outPath=os.path.join(args.basePath,'input_spike')
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    
    return args

#...!...!....................
def buildPayloadMeta(args):
    pd={}  # payload
    pd['raw_input_path']=args.inpPath
    pd['session_name']=args.sessionName
    pd['type']='experiment'
    md={ 'dataset':pd}
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
    for i in range(20):
        print(i,spikeT[i])
    return
    
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
    pmd=md['dataset']
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
    meaIdL=np.array(sorted(spike_dict),dtype=np.int16)  # here order of feature_id is settled
    maxFeat=len(meaIdL)
    # ... down select neurons
    
    if args.verb>1: print('RSD: meaID list:',meaIdL)
    pmd['num_feature']=len(meaIdL)
    print('mmm',meaIdL)
    
    spikeD={}
    spikeCntL=np.zeros(pmd['num_feature'],dtype=int)  # num spikes per  neuron
    maxTbin=0
    #dead_idL=[]
    j=0
    for k in meaIdL:
        rec=np.array(spike_dict[k])/args.time_rebin        
        spikeD[k]=rec.astype(int) # time-bin may repeat 
        spikeCntL[j]=len(rec)
        j+=1
        if len(rec)==0:
            #dead_idL.append(int(k))
            continue        
        mxTb=np.max(rec)
        if maxTbin< mxTb: maxTbin=mxTb
        #... check for smalest dist
        if j<5:qa_neuron(k,spikeD[k])
        
    pmd['num_time_bin']=int(maxTbin)+1
    pmd['max_time']=pmd['num_time_bin']/pmd['sampling_freq']

    bigD={'exp_feature_id':meaIdL, 'exp_spike_sum': spikeCntL}
    bigD['qa_avr_spike_freq']=spikeCntL/pmd['max_time']
    return  spikeD,bigD

#...!...!....................
def flatten_spike_data(bSpikeD,md,bigD): 
    pmd=md['dataset']
    nfeat=pmd['num_feature']
    ntime=pmd['num_time_bin']
    meaIdL=bigD['exp_feature_id']
    
    spikeA=np.zeros((nfeat,ntime),dtype=np.bool_)
    for k in range(nfeat):
        #print('k',k)
        fid=meaIdL[k]
        spikeA[k][bSpikeD[fid]]=True  # unpack spikes
                               
    timeV = np.linspace(0, ntime-1,  ntime,dtype=np.int32)
    bigD.update({'time_ms':timeV,'spikes_data':spikeA})
    return bigD
    

    
#...!...!....................
def mon_spike_freq(bSpikeD,md,bigD,twindow_sec=60.): 
    pmd=md['dataset']
    pmd['qa_twindow_sec']=twindow_sec
    nfeat=pmd['num_feature']
    ntime=pmd['num_time_bin']
    twindow_bin=int( pmd['sampling_freq'] * twindow_sec)
    num_window=1+ntime//twindow_bin
    meaIdL=bigD['exp_feature_id']
    freqA=np.zeros((nfeat,num_window),dtype=np.float32)
    for k in range(nfeat):
        fid=meaIdL[k]
        #fid=pmd['feature_id'][k]
        bSpikeV=bSpikeD[fid]//twindow_bin
        countV = np.bincount(bSpikeV)
        mxb=countV.shape[0]
        
        #print('ss',bSpikeV.shape,num_window,mxb);
        #print(freqV[:30])
        freqA[k][:mxb]=countV/twindow_sec
    expD['qa_spike_freq']=freqA




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
    binSpikeD,expD=read_spike_dict(expMD,args)
    #pprint(expMD)
    

    expD=flatten_spike_data(binSpikeD,expMD,expD)
    #... QA
    mon_spike_freq(binSpikeD,expMD,expD,twindow_sec=5.)   
  
    pprint(expMD)
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.spike.h5')
    write4_data_hdf5(expD,outF,expMD)
    print('   ./plot_expInput.py  --basePath $basePath  --inpName   %s  -p  b c d  -Y '%(expMD['short_name'] ))
    #print('   ./fit_uoiVar.py  --inpName   %s   \n'%(expMD['short_name'] ))
   


    
