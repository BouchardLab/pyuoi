#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import sys,os,hashlib
import numpy as np
import pickle
from pprint import pprint
import argparse
#...!...!..................
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--expPath",required=True,help="raw experimnetal data on CFS")
    
    parser.add_argument("--sessionName",  default='HET_80k_1',help='raw data session name')
    parser.add_argument("--dataPath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for any further data processing")
    parser.add_argument("--outName",  default=None,help='(optional) output file name - Is it needed?')

    # .... activity speciffic speciffic, 
    parser.add_argument('--time_rebin', default=100, type=int, help='rebin of raw time axis')
    parser.add_argument('--freqRange', default=[0.1,30], type=float, nargs=2,help='rebin of raw time axis')
    
    args = parser.parse_args()
    
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
        
    return args

#...!...!....................
def buildBioMeta(args):
    pd={}  # payload
    pd['raw_bioexp_path']=args.expPath
    pd['session_name']=args.sessionName
    pd['culture_type']='my culture 77'
    pd['recording_date']='19630417'
    pd['chip_name']='M12345'
    pd['run_number']='010203'
    pd['well_no']='Well1234'

    sel={'freq_range':args.freqRange}
    md={ 'bioexp':pd,'selector':sel}
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash']=myHN
    if args.outName==None:
        md['short_name']='%s-%s'%(args.sessionName,md['hash'])
    else:
        md['short_name']=args.outName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md

def read_spike_dict(md,args):
    pmd=md['bioexp']
    inpF=os.path.join(args.expPath,args.sessionName,'spike_dict.pkl')
    print('inpF:',inpF)
    assert os.path.exists(inpF)
    # Load the dictionary from the .pkl file
    with open(inpF, "rb") as f:
        spike_dict = pickle.load(f)

    raw_sampling_freq=10000  # Hz, number from Roy
    assert raw_sampling_freq%args.time_rebin==0 
    pmd['sampling_freq'] =raw_sampling_freq/args.time_rebin   
    
    # neuron ID  MEA chip
    meaIdL=np.array(sorted(spike_dict))  # here order of feature_id is settled
    maxFeat=len(meaIdL)
    
    if args.verb>1: print('RSD: meaID list:',meaIdL)
    pmd['num_feature']=len(meaIdL)
     
    spikeT={}  # spike times
    spikeCntL=np.zeros(pmd['num_feature'],dtype=int)  # num spikes per  neuron
    maxTbin=0

    j=0
    for k in meaIdL:
        rec=np.array(spike_dict[k])/args.time_rebin        
        spikeT[k]=rec.astype(int) # time-bin may repeat 
        spikeCntL[j]=len(rec)
        j+=1
        if len(rec)==0:
             continue        
        mxTb=np.max(rec)
        if maxTbin< mxTb: maxTbin=mxTb
        
    pmd['num_time_bin']=int(maxTbin)+1
    pmd['max_time']=pmd['num_time_bin']/pmd['sampling_freq']
    chanFreq=spikeCntL/  pmd['max_time']
    rawD={'spikeT':spikeT,'chanFreq':chanFreq,'chanID':meaIdL}
    return  rawD

#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    args=commandline_parser()
    np.set_printoptions(precision=3)
    bioMD=buildBioMeta(args)
    
    # read raw data
    rawD=read_spike_dict(bioMD,args)
    pprint(bioMD)
    print('rawD',sorted(rawD))
    
