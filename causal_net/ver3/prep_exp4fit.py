#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
prepares input for UoI fit experimental data

HD5 arrays contain input and output

Use case .pkl *****:

basePath=/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp2/
inputPath=/global/cfs/cdirs/m2043/causal_inference/DIV13

ses=HET_80k_1 ; ses2=${ses}
./prep_exp4fit.py --sessionName $ses --outName $ses2  --basePath $basePath 

Use case .npy *****:

basePath=/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp2/
inpPath=/global/cfs/cdirs/m2043/causal_inference/Canine_Organoids_PVS/Analysis/250619/M08020/Network/
ses=000093/well001; ses2=250619_M08020_run93_well1


./prep_exp4fit.py --sessionName $ses --outName $ses2 --inpExt npy --basePath $basePath --inpPath $inpPath

Decoding the name:
Causal inference – project name
Canine_organoids_PVS – Type of the Culture
Analysis – its analysis folder
250619 – recording data 
M08020 – Plate/ chip name
Network – type of recording
000093 – run number
Well000 -> is the well number ( indexed at 0)



'''
import sys,os,hashlib
import numpy as np
import pickle
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from Util_CausalNet import compute_spike_moments

import time

import argparse
#...!...!..................
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--inpPath",default='/global/cfs/cdirs/m2043/causal_inference/DIV13',help="raw input data")
    
    parser.add_argument("--sessionName",  default='HET_80k_1',help='raw data session name')
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")    
    parser.add_argument("--outName",  default=None,help='(optional) output file name')
    parser.add_argument("--inpExt",  default='pkl',choices=("pkl", "npy"),help='type of input: pkl or npy')
    parser.add_argument('--filterData' , default=[],  nargs='+',   type=float, help='list: freqLo  freqHi ')

   
    args = parser.parse_args()    
    args.outPath=os.path.join(args.basePath,'input_fitter')
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    assert len(args.filterData) in [0,2]
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
    dsm=md['dataset']
    inpF=os.path.join(args.inpPath,args.sessionName,'spike_dict.pkl')
    print('inpF:',inpF)
    assert os.path.exists(inpF)
    # Load the dictionary from the .pkl file
    with open(inpF, "rb") as f:
        spike_dict = pickle.load(f)

    # hardcoded initial timing and binning , must match Roy's data
    raw_sampling_freq=10000  # Hz
    time_rebin=10  #
    
    assert raw_sampling_freq%time_rebin==0
    dt=time_rebin/raw_sampling_freq
    dsm['step_duration']=dt
    
    # neuron ID  MEA chip
    meaIdV=np.array(sorted(spike_dict),dtype=np.int16)  # here order of feature_id is settled
    maxFeat=meaIdV.shape[0]
    # ... down select neurons
    
    if args.verb>1: print('RSD1: meaID list:',meaIdV)
    dsm['num_feature']=maxFeat
        
    spikeD={}
    spikeCntV=np.zeros(maxFeat,dtype=int)  # num spikes per  neuron
    maxTbin=0
    
    j=0
    for k in meaIdV:
        rec=np.array(spike_dict[k])/time_rebin        
        spikeD[k]=rec.astype(int) # time-bin may repeat 
        spikeCntV[j]=len(rec)
        j+=1
        if len(rec)==0:
            #dead_idL.append(int(k))
            continue        
        mxTb=np.max(rec)
        if maxTbin< mxTb: maxTbin=mxTb
        #... check for smalest dist
        #if j<5:qa_neuron(k,spikeD[k])
        
    dsm['num_time_steps']=int(maxTbin)+1
    dsm['max_time']=dsm['num_time_steps']*dt
    dsm['spikes_rate']=float(np.sum(spikeCntV)/dsm['max_time'])
    
    bigD={'exp_feature_id':meaIdV, 'exp_spike_sum': spikeCntV}
    bigD['qa_avr_spike_freq']=spikeCntV/dsm['max_time']
    return  spikeD,bigD


#...!...!....................
def read_spike_numpy(md,args):
    dsm=md['dataset']
    inpF=os.path.join(args.inpPath,args.sessionName,'spike_times.npy')
    print('inpF:',inpF)
    assert os.path.exists(inpF)

    raw = np.load(inpF, allow_pickle=True)
        
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        data = raw.item()
        #print("Unwrapped 0-d object array; now data is", type(data))
    else:
        data=raw
    assert isinstance(data, dict)
    keys = list(data.keys())
    print(f"\nDetected dict with {len(keys)} keys.")
    print('Sample keys:',keys[:10],'...', keys[-10:])
    
    # neuron ID  MEA chip
    meaIdV=np.array(keys,dtype=np.int16)  # here order of feature_id is settled        
    maxFeat=meaIdV.shape[0]
    # ... down select neurons
    
    # hardcoded initial timing and binning , must match Roy's data
    raw_sampling_freq=10000  # Hz
    time_rebin=10  #
    
    assert raw_sampling_freq%time_rebin==0
    dt=time_rebin/raw_sampling_freq
    dsm['step_duration']=dt
    
    if args.verb>1: print('RSD2: meaID list:',meaIdV)
    dsm['num_feature']=maxFeat
        
    spikeD={}
    spikeCntV=np.zeros(maxFeat,dtype=int)  # num spikes per  neuron
    maxTbin=0

    for i, k in enumerate(meaIdV):
        v = data[k]
        rec = np.asarray(v)/dt

        spikeD[k]=rec.astype(int) # time-bin may repeat 
        spikeCntV[i]=len(rec)
        
        if len(rec)==0:  continue        
        mxTb=np.max(rec)
        if maxTbin< mxTb: maxTbin=mxTb

    dsm['num_time_steps']=int(maxTbin)+1
    dsm['max_time']=dsm['num_time_steps']*dt
    dsm['spikes_rate']=float(np.sum(spikeCntV)/dsm['max_time'])

    bigD={'exp_feature_id':meaIdV, 'exp_spike_sum': spikeCntV}
    bigD['qa_avr_spike_freq']=spikeCntV/dsm['max_time']
    return  spikeD,bigD

#...!...!....................
def flatten_spike_data(bSpikeD,md,bigD): 
    pmd=md['dataset']
    nfeat=pmd['num_feature']
    ntime=pmd['num_time_steps']
    meaIdL=bigD['exp_feature_id']
    
    spikeA=np.zeros((nfeat,ntime),dtype=np.int32)
    for k in range(nfeat):
        #print('k',k)
        fid=meaIdL[k]
        spikeA[k][bSpikeD[fid]]=True  # unpack spikes
                               
    timeV = np.linspace(0, ntime-1,  ntime,dtype=np.int32)
    bigD.update({'time_ms':timeV,'spikes_data':spikeA})
    return bigD

#...!...!....................
def filter_data(args,md,bigD):
    meaIdL = bigD['exp_feature_id']
    spikeDV = bigD['spikes_data']  # fixed typo: was expD, should be bigD
    dsm = md['dataset']
    dt = dsm['step_duration']
    nFeat = dsm['num_feature']
    freqLo, freqHi = args.filterData

    print('\nFilterData: nFeat=%d dt=%.3f  freqLo/Hi= %.1f %.1f' % (nFeat, dt, freqLo, freqHi))

    accepted = []
    rejected = []
    freqMin=9999; freqMax=-1
    xxx
    for k in range(nFeat):
        fid = meaIdL[k]
        spikeV = spikeDV[fid]
        total_spikes = spikeV.sum()
        total_time = len(spikeV) * dt  # in seconds
        freq = total_spikes / total_time if total_time > 0 else 0.0  # Hz
        if freqMin> freq : freqMin= freq 
        if freqMax< freq : freqMax= freq 
        #print('Neuron %s: total_spikes=%d, total_time=%.2f s, freq=%.2f Hz' % (fid, total_spikes, total_time, freq))

        if freqLo <= freq <= freqHi:
            accepted.append(fid)
        else:
            rejected.append(fid)
    print('scan %d neurons, freq min/max= [%.2f, %.2f] Hz'%(nFeat,freqMin,freqMax))
    print('\nAccepted %d of %d neuron IDs in freq in [%.1f, %.1f] Hz):' % (len(accepted),nFeat,freqLo, freqHi))
    print(accepted)
    print('\nRejected neuron IDs (freq outside range):')
    print(rejected)

    

#...!...!....................
def mon_spike_freq(bSpikeD,md,bigD,twindow_sec=60.): 
    pmd=md['dataset']
    pmd['qa_twindow_sec']=twindow_sec
    nfeat=pmd['num_feature']
    ntime=pmd['num_time_steps']
    twindow_bin=int(  twindow_sec/pmd['step_duration'])
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
    if args.inpExt=='pkl':
        binSpikeD,expD=read_spike_dict(expMD,args)
    if args.inpExt=='npy':
        binSpikeD,expD=read_spike_numpy(expMD,args)
    #pprint(expMD)
    

    expD=flatten_spike_data(binSpikeD,expMD,expD)
    if len(args.filterData)>0:
        filter_data(args,expMD,expD)
    
    #... QA
    mon_spike_freq(binSpikeD,expMD,expD,twindow_sec=5.)   
    expD['qa_spike_moments']=compute_spike_moments(expD['spikes_data'], maxRebin=11,maxTime=300_000, verb=1)
    
  
    pprint(expMD)
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.spikes.h5')
    write4_data_hdf5(expD,outF,expMD)
    print('   ./plot_fitInput.py  --basePath $basePath  --inpName   %s  -p e   -Y '%(expMD['short_name'] ))
    
   


    
