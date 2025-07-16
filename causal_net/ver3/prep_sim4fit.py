#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
prepares input for UoI fit simulated Dale-generator data

HD5 arrays contain input and output

Use case:

*** on PM
basePath=/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp2/

./prep_simInput.py --sessionName $ses --outName $ses2  --basePath $basePath
./plot_features.py   --basePath $basePath --inpName   $ses2 -p  d -Y


'''
import sys,os,hashlib
import numpy as np
import pickle
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from Util_CausalNet import daleMatrix_index_partition
import argparse


#...!...!..................
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--basePath",default='out',help="head dir for set of experimentst")
    parser.add_argument("--simName",  required=True,help='name of input data')
    parser.add_argument("--outName",  default=None,help='output name')
    parser.add_argument("--time_start", type=int, default=5,help="start time (time steps)")
    parser.add_argument("--time_rebin", type=int, default=1,help="reduce time resolution by the rebin factor")
    
    args = parser.parse_args()
    args.inpPath=os.path.join(args.basePath,'gen_dale')
    args.outPath=os.path.join(args.basePath,'input_fitter')
    
    if args.verb>0:
        print( 'myArg-program:',parser.prog)
        for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
        print('',flush=True)
    
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def rebin_sum(V, nReb):
    nFeat, nT = V.shape
    nBins = nT // nReb  # number of complete bins
    nT_clip = nBins * nReb
    V_clip = V[:, :nT_clip]  # drop last incomplete bin if needed
    V_rebinned = V_clip.reshape(nFeat, nBins, nReb).sum(axis=2)
    return V_rebinned

#...!...!....................
def format_simNetActivity(inpD,md):
    nReb=args.time_rebin
    md['selector']=sel={}    
    sel['input_name']=args.simName
    sel['time_start']=args.time_start
    sel['time_rebin']=nReb

    md['hash']=inpMD['hash']
    myName=inpMD['short_name']
    if nReb>1:
        tag='_%dms'% sel['time_rebin']
        md['hash']+=tag
        myName+=tag
    if args.outName!=None:
        md['short_name']=args.outName
    else:
        md['short_name']=myName
        
    stateV=inpD['simu_state']
    spikeV=inpD['simu_spikes']

    #.... any data transformation goes here ....
    #.... clip data
    tL=args.time_start
    assert tL < stateV.shape[1]
    stateV=stateV[:,tL:].astype(np.int32)
    spikeV=spikeV[:,tL:].astype(np.float32)

    if nReb>1:
        stateV=rebin_sum(stateV,sel['time_rebin'])
        spikeV=rebin_sum(spikeV,sel['time_rebin'])
        dsm=md['dataset']
        dsm['max_rate_per_step']=int(np.max(spikeV))
        dsm['step_duration']*=nReb
        dsm['num_time_steps']=stateV.shape[1]
        print('Data rebinned by ',nReb)
    
    outD={}
    outD['stateVec_data']=stateV.astype(np.float32)
    outD['spikes_data']=spikeV.astype(np.int32)
    outD['true_network_matrix']=inpD['true_network_matrix'].astype(np.float16)
    

    # Compute true_matrix_5index for Dale matrix partitioning
    Mt = inpD['true_network_matrix'].T

    Ldia, Lexc, Lzexc, Linh, Lzinh = daleMatrix_index_partition(Mt)
    md['dale_truth']['5index'] = {
        'diag': Mt[Ldia[0]].shape[0],
        'exc_nonzero': Mt[Lexc[0]].shape[0],
        'exc_zero': Mt[Lzexc[0]].shape[0],
        'inh_nonzero': Mt[Linh[0]].shape[0],
        'inh_zero': Mt[Lzinh[0]].shape[0]
    }


    return md,outD
    
 
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    args=commandline_parser()
    np.set_printoptions(precision=4)
    
    inpF=os.path.join(args.inpPath,args.simName+'.simNet.h5')
    inpD,inpMD=read4_data_hdf5(inpF)

    if 1: # patch for old data
        if 'dale_truth' in inpMD['dale_truth']: inpMD['dale_truth']['dale_name']=inpMD['dale_truth'].pop('dale_truth')

    expMD,expD=format_simNetActivity(inpD,inpMD)
    
    pprint(expMD)
   
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.spikes.h5')
    write4_data_hdf5(expD,outF,expMD)

    nReb=args.time_rebin
    txt1=''
    if nReb>1: txt1='  --max_lag_bin %d --fit_start_bin %d '%(400//nReb,1+16//nReb)
    
    
    print('   ./plot_fitInput.py  --basePath $basePath   --inpName   %s  -p e  %s -Y '%(expMD['short_name'],txt1 ))
    print(' shifter  --image nersc/pytorch:25.02.01 python fit_xcorrelogram.py  --data_path ${basePath}//input_fitter   --file_name   %s.spikes.h5   --max_lag 10 --n_shuffles 500  --sparsity 0.5  \n'%(expMD['short_name'] ))

    print('  srun -n512 --distribution=block:block shifter python  fit_uoiVar.py  --basePath $basePath   --inpName   %s  --num_admm 32  --time_range 0 10_000  \n'%(expMD['short_name'] ))
   
    
