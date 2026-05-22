#!/usr/bin/env python3
"""
Preprocessing pipeline for experimental neural data from Roy/Mandar laboratory.

This script processes raw experimental neural recordings into standardized format
for connectivity analysis. The preprocessing pipeline includes:
- Raw data loading and format conversion
- Temporal binning and spike count extraction  
- Data quality assessment and filtering
- Metadata extraction and session identification
- Output formatting for downstream analysis tools

Session naming convention:
- B6J: cell line name
- 250619: recording date (YYMMDD)
- M08020: chip identifier  
- 000093: run number
- Well000: well number

The script generates .spikes.npz files with standardized spike count matrices
and associated metadata for further analysis.

Usage:
    ./prep_bioexp.py --sessionName B6J_250619_M08020_000093_Well000 --inputPath /path/to/raw/data/
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
import sys,os,hashlib
import numpy as np
import pickle
from pprint import pprint
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz

import argparse
#...!...!..................
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--expPath",required=True,help="raw experimnetal data on CFS")
    
    parser.add_argument("--sessionName",  default='celllinename/dateofrecording/chipID/Assaytype/runnumber/wellnumber',help='raw data session name')
    parser.add_argument("--dataPath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for any further data processing")
    parser.add_argument("--shortName",  default=None,help='(optional) output file name - Is it needed?')

    # .... activity speciffic speciffic, 
    parser.add_argument('--samp_freq', default=100, type=int, help='sets binning of time axis')

    parser.add_argument('--freqRange', default=[1.,50], type=float, nargs=2,help='rebin of raw time axis')
    
    args = parser.parse_args()
    
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
        
    return args

#...!...!....................
def buildBioMeta(args):
    pd={}  # payload
    pd['raw_bioexp_path']=args.expPath
    pd['session_name']=args.sessionName
    txtL=args.sessionName.split('/')
    #print('tt',txtL); aa
    pd['cell_line_name']=txtL[0]
    pd['recording_date']=txtL[1]
    pd['chip_ID']=txtL[2]
    pd['run_num']=txtL[4]
    pd['well_num']=txtL[5]

    sel={'freq_range':args.freqRange}
    md={ 'bioexp':pd,'data_selector':sel}
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash']=myHN
    if args.shortName==None:
        md['short_name']='%s-%s'%(pd['recording_date'],md['hash'])
    else:
        md['short_name']=args.shortName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md


def read_spike_npy(md,args):
    pmd=md['bioexp']
    inpF=os.path.join(args.expPath,args.sessionName,'spike_times.npy')
    print('inpF:',inpF)
    assert os.path.exists(inpF)
    # Load the dictionary from the .npy file
    spike_dict = np.load(inpF, allow_pickle=True).item()

    print('spike_dict',spike_dict);ok
   
    pmd['sampling_freq'] =args.samp_freq
    
    # neuron ID  MEA chip
    meaIdL=np.array(sorted(spike_dict))  # here order of feature_id is settled
    maxFeat=len(meaIdL)
    
    if args.verb>1: print('RSN: meaID list:',meaIdL)
    pmd['num_feature']=len(meaIdL)
     
    spikeT={}  # spike times
    spikeCntL=np.zeros(pmd['num_feature'],dtype=int)  # num spikes per  neuron
    maxTbin=0

    j=0
    for k in meaIdL:
        rec=np.array(spike_dict[k])*args.samp_freq
        #print(rec[:100],len(rec)) 
        spikeT[k]=rec.astype(int) # time-bin may repeat 
        spikeCntL[j]=len(rec)
        j+=1
        if len(rec)==0:
             continue        
        mxTb=np.max(rec)
        if maxTbin< mxTb: maxTbin=mxTb
        #exit(0)    
    pmd['num_time_bin']=int(maxTbin)+1
    pmd['max_time']=pmd['num_time_bin']/pmd['sampling_freq']
    chanFreq=spikeCntL/  pmd['max_time']
    rawD={'spikeT':spikeT,'chanFreq':chanFreq,'MEA_idx':meaIdL}
    return  rawD


#...!...!....................
def unroll_bioexp(rawD,bioMD): 
    pmd=bioMD['bioexp']
    sel=bioMD['data_selector']
    frLo, frHi = sel['freq_range']
    print('frLo, frHi',frLo, frHi)
    assert frLo < frHi
    chanFreq = np.asarray(rawD['chanFreq'], dtype=float)    
    # vectorized boolean mask for channels within (frLo, frHi) range
    freqMask = (chanFreq >= frLo) & (chanFreq <= frHi)
    sel['num_drop_neur_lo_hi_freq']=[ int(np.sum(chanFreq < frLo)),  int(np.sum(chanFreq > frHi)) ] 
    print('freqMask all=%d , passed=%d'%(freqMask.shape[0],np.sum(freqMask)))
    #print(sel);aaa
    # --- drop channles out of freq range
    chanFreq=rawD['chanFreq'][freqMask]
    MEA_idx=rawD['MEA_idx'][freqMask]

   # .... REMAP MATRICES TO FREQUENCY-SORTED ORDER (PRIMARY INDEX)
    neur_freqIdx = np.argsort(chanFreq)  # indices that sort chanFreq by value
    neur_revFreqIdx = np.empty(len(neur_freqIdx), dtype=int)  # natural_index → freq_sorted_position
    neur_revFreqIdx[neur_freqIdx] = np.arange(len(neur_freqIdx))
    
    #--- reorder channles by frequency
    chanFreq=chanFreq[neur_freqIdx]
    MEA_idx=MEA_idx[neur_freqIdx]
    print('chanFreq',chanFreq[:5],'...',chanFreq[-5:],'Hz')

    # create spike matrix: rows=time bins, cols=accepted channels
    ntime=pmd['num_time_bin']
    nchan=MEA_idx.shape[0]
    spikes2D=np.zeros((ntime,nchan),dtype=np.int32)
    spikeT=rawD['spikeT']
    for ic, ch in enumerate(MEA_idx):
        tV=spikeT[ch]
        if len(tV)==0:  continue
        # tV holds time-bin indices where this channel fired one or more spikes
        # bincount returns a length-ntime vector with spike counts per bin (zeros elsewhere)
        # minlength=ntime guarantees the vector spans the full recording duration
        cnt=np.bincount(tV, minlength=ntime)
        spikes2D[:,ic]=cnt.astype(np.int32)
    print('spikes2D shape',spikes2D.shape)

    # keep handy in meta for downstream
    sel['num_chan']=nchan
    sel['max_spike_per_bin']=int(np.max(spikes2D))

    Y_uchar = np.clip(spikes2D, 0, 255).astype(np.uint8)
    spikeD={'spikes':Y_uchar,
          'single_rates':chanFreq
            }

    bioD={ }
    bioD['neur_freqIdx']=neur_freqIdx
    bioD['neur_revFreqIdx']=neur_revFreqIdx
    bioD['MEA_idx']=MEA_idx

    #.... compute neural statistics for spikeMD
    num_neurons = nchan
    avg_rate = float(np.mean(chanFreq))
    std_rate = float(np.std(chanFreq))
    median_rate = float(np.median(chanFreq))
    min_rate = float(np.min(chanFreq))
    max_rate = float(np.max(chanFreq))
    
    # Compute Fano factor (variance/mean) for each neuron
    mean_counts_per_bin = np.mean(spikes2D, axis=0)
    spike_variance = np.var(spikes2D, axis=0)
    fano_factor = np.divide(spike_variance, mean_counts_per_bin, out=np.zeros_like(spike_variance), where=mean_counts_per_bin != 0)
    avg_fano = float(np.mean(fano_factor))
    std_fano = float(np.std(fano_factor))
    
    # Print summary statistics
    print('Neural Statistics Summary:')
    print('num neurons: %d, Avg Rate= %.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_neurons, avg_rate, std_rate, avg_fano, std_fano))
    print('Median rate  %.2f Hz' % median_rate)
    
    #.... extract spikeMD for fitter
    spikeMD={'time_step_sec': 1./pmd['sampling_freq'],
             'provenance': {'experiment_name':bioMD['short_name']},
             'poisson_eta_clip': 5, # expected by fitter
             'data_type':'bioExp', 'num_neurons': num_neurons }

    bioMD['rate_summary']={
        'avg_spike_rate': avg_rate,
        'std_spike_rate': std_rate,
        'avg_fano_factor': avg_fano,
        'std_fano_factor': std_fano,
        'median_spike_rate': median_rate,
        'min_spike_rate': min_rate,
        'max_spike_rate': max_rate
    }
    return bioD,spikeD,spikeMD
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    args=commandline_parser()
    np.set_printoptions(precision=3)
    bioMD=buildBioMeta(args)
    
    # read raw data
    #rawD=read_spike_dict(bioMD,args)
    rawD=read_spike_npy(bioMD,args)

    #.... filter & unroll data
    bioD,spikeD,spikeMD=unroll_bioexp(rawD,bioMD)

    #...... WRITE   OUTPUT .........
    outFt = os.path.join(args.dataPath, bioMD['short_name'] + '.bioExp.npz')
    write_data_npz(bioD, outFt, metaD=bioMD)
    if args.verb>2:
        print('\n bioD:',sorted(bioD))
        pprint(bioMD)
  
    outFs = outFt.replace('.bioExp.','.spikes.')
    write_data_npz(spikeD, outFs, metaD=spikeMD)
    if args.verb>2:  
        print('\nspikeD:',sorted(spikeD))
        pprint(spikeMD)

    print("\nNext step command:")
    print('   ./view_bioexp.py  --dataPath $dataPath  --dataName   %s  -p  a b   -T 0 3550  '%(bioMD['short_name'] ))
    print("  ./fit_lassoPoisson.py  --dataPath $dataPath  --dataName %s  --num_epochs  10 " % bioMD['short_name'] )
    print(" ./fitLasso4GPU.sh  --dataPath $dataPath  --dataName %s  --num_epochs  200 " % bioMD['short_name'] )
    print("  ./bootsFit.sh --dataName  %s  --num_epochs  250 --dropDataFrac 0.5 --num_bootstraps 7  " % bioMD['short_name'] )

    print("   ./selectEdges_FDR.py  --dataName %s  --num_bootstraps 6 10 -p a c d  " % bioMD['short_name'] )
    
    print('    --dataPath '+args.dataPath)
   

    
