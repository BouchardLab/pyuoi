#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

ses=HET_80k_1 ; ses2=${ses}_samp1kHz
./prep_input.py --sessionName $ses --outName $ses2  --basePath /global/cfs/cdirs/m2043/causal_inference/DIV13
./plot_expInput.py  --inpName   $ses2 -p  e -Y

Plot options:
 -p a : input features (individual spike traces)
 -p b : dense features (multiple spike traces)  
 -p c : global QA (frequency analysis)
 -p d : detailed QA (2D frequency heatmaps)

'''

import os
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from time import time
from pprint import pprint
import numpy as np
from PlotterExperiment import Plotter


from Util_CausalNet import compute_mean_crosscov_fastV2, fit_exponent_weighted,  compute_mean_autocovV2, compute_spike_moments

import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcde-string listing shown plots: a=input_features, b=dense_features, c=global_qa, d=detailed_qa, e=spike_moments_analysis")
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")         
    parser.add_argument("--basePath",default='out',help="head dir for set of experimentst")
    parser.add_argument('--time_range' , default=[0, 300_000],  nargs=2,   type=int, help='cov  data time range (bins)')
    parser.add_argument('--max_lag' , default=400,   type=int, help='cov lag (bins)')
    parser.add_argument("--inpName",  default='exp_62a21daf',help='IBMQ experiment name assigned during submission')
    
    args = parser.parse_args()
    # make arguments  more flexible
    args.fit_start_bin=50
    args.dataPath=os.path.join(args.basePath,'input_fitter' )
    args.outPath=os.path.join(args.basePath,'postproc')
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    return args


#...!...!....................

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
                    
    inpF=args.inpName+'.spikes.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            print(expD)
        stop2
        
    timeStep=expMD['dataset']['step_duration']
    spikes=expD['spikes_data'][:, args.time_range[0]: args.time_range[1]]

    print('M: sample size:',spikes.shape, args.time_range)
    
    if  'e' in args.showPlots: # fit Cox‐process exponential decay
        max_lag=args.max_lag  # (bins)
        fit_start=args.fit_start_bin; N_total = spikes.shape[1] 
        T0=time()

        expD['qa_spike_moments']=compute_spike_moments(expD['spikes_data'], maxRebin=11,maxTime=None, verb=1)

        #1)  auto/corss-correlation
        autoCovData = compute_mean_autocovV2(spikes,  max_lag=max_lag)
        crosCovData = compute_mean_crosscov_fastV2(spikes, max_lag=max_lag)
        print('M:  computed in elaT=%.1f sec'%(time() -T0))
        print('ss1',spikes.shape,autoCovData.shape)
        # Merge times and covData into one 2D vector
        expD['auto_cov_data'] = autoCovData
        expD['cross_cov_data'] =  crosCovData
        
        #  fit Cox‐process exponential decay
        #expMD['auto_cov_fit']=fit_exponent_weighted( autoCovData,  N_total, timeStep, fit_start=fit_start)
        #expMD['cross_cov_fit']=fit_exponent_weighted( crosCovData, N_total, timeStep, fit_start=fit_start )
        expMD['cov_spikes_data_shape']=list(spikes.shape)

        
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']+'_exp'
    expMD['plot']={}
    
    #expMD['plot']['time_rangeLR']=[1200,1800]
   
    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.input_features(expD,expMD,figId=1,mxFeat=6)
    if 'b' in args.showPlots:
        plot.input_features_dense(expD,expMD,figId=2,mxFeat=9)

    if 'c' in args.showPlots:
        plot.global_qa(expD,expMD,figId=3)
        
    if 'd' in args.showPlots:
        plot.detailed_qa(expD,expMD,figId=4)

    if 'e' in args.showPlots:
        
        plot.fano_and_cov_fit( expD,expMD,figId=6)
 
    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
