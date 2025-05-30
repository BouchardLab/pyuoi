#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Post-process UoI-VAR fitted model and analyze results.
Usage: ./postproc_uoiVar.py
'''

import os
import argparse
from time import time
import numpy as np
from pprint import pprint

from toolbox.Util_H5io4 import write4_data_hdf5, read4_data_hdf5
from toolbox.Util_CausalNet import print_dale_matrix, daleMatrix_index_partition, residual_stats
from PlotterModelFit import Plotter

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0,1,2,3,4], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p","--showPlots", default='a b', nargs='+',help="abcd-string listing shown plots")
    parser.add_argument("-Y","--noXterm", dest='noXterm', action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("--basePath",default='out',help="head dir for any results")
    parser.add_argument('-e',"--expName", default='fit-b7018a',help='UoI-VAR fitted model')
    parser.add_argument('-m','--max_feature', default=15, type=int, help='max num of analyzed features')

    args = parser.parse_args()
    args.modelPath=os.path.join(args.basePath,'model_uoi')
    args.outPath=os.path.join(args.basePath,'postproc')
    args.showPlots=''.join(args.showPlots)
  
    print('myArg-program: %s'%(parser.prog))
    for arg in vars(args): print('myArg: %s %s'%(arg, getattr(args, arg)))
    
    assert os.path.exists(args.modelPath)
    assert os.path.exists(args.outPath)
    return args

def postproc_fit(bigD,md):
    Mt=bigD['true_network_matrix'].T
    lag=0
    Mf=bigD['fit_A_model'][lag].T
    Ldia,Lexc,Lzexc,Linh,Lzinh = daleMatrix_index_partition(Mt)
    print('PPF: dale partition 1st elem size: diag:%s  exc:%s  zexc:%s  inh:%s  zinh:%s'%(Mt[Ldia[0]].shape, Mt[Lexc[0]].shape, Mt[Lzexc[0]].shape, Mt[Linh[0]].shape, Mt[Lzinh[0]].shape))

    # Add matrix shapes to metadata
    md['matrix_shape']={
        'diag': Mt[Ldia[0]].shape,
        'exc': Mt[Lexc[0]].shape,
        'zexc': Mt[Lzexc[0]].shape,
        'inh': Mt[Linh[0]].shape,
        'zinh': Mt[Lzinh[0]].shape
    }

    Ydia=np.stack((Mt[Ldia],Mf[Ldia]), axis=1)[1:]
    Yexc=np.stack((Mt[Lexc],Mf[Lexc]), axis=1)
    Yinh=np.stack((Mt[Linh],Mf[Linh]), axis=1)
    Yzexc=Mf[Lzexc][Mf[Lzexc]!=0]
    Yzinh=Mf[Lzinh][Mf[Lzinh]!=0]

    print('diag shape: %s'%(str(Ydia.shape)))
    print('Yexc,z shape: %s %s'%(str(Yexc.shape),str(Yzexc.shape)))
    print('Yinh,z shape: %s %s'%(str(Yinh.shape),str(Yzinh.shape)))
  
    bigD['post_Ydia']=Ydia
    bigD['post_Yexc']=Yexc
    bigD['post_Yzexc']=Yzexc
    bigD['post_Yinh']=Yinh
    bigD['post_Yzinh']=Yzinh

    pof=md['post_fit_residual']={}
    stats,Xp,Yp = residual_stats(Ydia)
    pof['diag']=stats
    bigD['post_Rdia']=Yp

    stats,Xp,Yp = residual_stats(Yexc)
    pof['exc']=stats
    bigD['post_Rexc']=Yp
    
    stats,Xp,Yp = residual_stats(Yinh)
    pof['inh']=stats
    bigD['post_Rinh']=Yp
    #pprint(pof)

def nice_print_model(bigD,md,mxFeat=None):
    pmd=md['payload']
    sem=md['selector']
    fim=md['fit_uoi']
    lag=fim['lag_depth']
    ntime,nfeat=fim['data_shape']
    inpN=sem['input_name']
    print('Postproc UoI-VAR input=%s nfeat=%d ntime=%d lag=%d fitTime=%.1f sec ranks=%d'%(inpN,nfeat,ntime,lag,fim['fit_time'],fim['num_rank']))
    
    AV=bigD['fit_A_model']
    if mxFeat!=None: nfeat=min(mxFeat,nfeat)
    
    for il in range(lag):
        print('\nA_model[%d]'%(il))
        A=AV[il]
        print_dale_matrix(A,nfeat)    
 
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
                    
    inpF=args.expName+'.fitUoI.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.modelPath,inpF))
    
    if args.verb>=2:
        print('M:expMD:')
        pprint(expMD)

    nice_print_model(expD,expMD,mxFeat=args.max_feature)
    postproc_fit(expD,expMD)
  
    outF=os.path.join(args.outPath,expMD['short_name']+'.post.h5')
    write4_data_hdf5(expD,outF,expMD)

    args.prjName=expMD['short_name']
    plot=Plotter(args)  
  
    if 'a' in args.showPlots: plot.A_matrix(expD,expMD,figId=1,lag=0)
    if 'b' in args.showPlots: plot.Aper_row(expD,expMD,figId=2,lag=0)
    if 'c' in args.showPlots: plot.A_matrix(expD,expMD,figId=3,lag=-1)
    if 'd' in args.showPlots: plot.Aper_row(expD,expMD,figId=4,lag=-1)
    if 'e' in args.showPlots: plot.weigh_correl(expD,expMD,figId=5)
        
    plot.display_all()
    print('M:done')
   
