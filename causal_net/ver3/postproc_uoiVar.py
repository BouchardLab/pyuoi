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
from Util_CausalNet import print_dale_matrix, daleMatrix_index_partition, residual_stats
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

def eval_fit(md,bigD):
    Ydiag=bigD['post_Ydiag']
    Yexc=bigD['post_Yexc']
    Yzexc=bigD['post_Yzexc']
    Yinh=bigD['post_Yinh']
    Yzinh=bigD['post_Yzinh']
    resExc=Yexc[:,0]-Yexc[:,1]
    fitExc=Yexc[:,1]
    resInh=Yinh[:,0]-Yinh[:,1]
    fitInh=Yinh[:,1]
    resDiag=Ydiag[:,0]-Ydiag[:,1]
    fitDiag=Ydiag[:,1]
    
    mxs=md['dale_truth']['5index'] 
    pfr=md['post_fit_residuals']

    fev=md['fit_eval']={}
    fev['names']={'fit':md['short_name'],'input':md['selector']['input_name']}

    #.....diagonal
    fevd=fev['diag']={'num_true':mxs['diag'], 'num_neg':np.sum(fitDiag < 0),'std_dev':np.std(resDiag),'rho':pfr['diag']['rho'] } 

    
    feve=fev['exc']={} #..... excitatory
    feve['true_nonzero']={'num_pos':np.sum(fitExc > 0), 'num_neg':np.sum(fitExc < 0), 'num_zero':np.sum(fitExc == 0),'std_dev':np.std(resExc),'num_true':mxs['exc_nonzero'],'rho':pfr['exc']['rho']}
    feve['true_zero']={'num_nonzero':Yzexc.size,'std_dev':np.std(Yzexc),'mean':np.mean(Yzexc),'num_true':mxs['exc_zero']}
    
    fevi=fev['inh']={} #..... inhibitory
    fevi['true_nonzero']={'num_pos':np.sum(fitInh > 0), 'num_neg':np.sum(fitInh < 0), 'num_zero':np.sum(fitInh == 0),'std_dev':np.std(resInh),'num_true':mxs['inh_nonzero'],'rho':pfr['inh']['rho']}
    fevi['true_zero']={'num_nonzero':Yzinh.size,'std_dev':np.std(Yzinh),'mean':np.mean(Yzinh),'num_true':mxs['inh_zero']}

def postproc_fit(bigD,md):
    pmd=md['dataset']
    Mt=bigD['true_network_matrix'].T
    lag=0
    fac=pmd['tau_response']/pmd['step_duration']
    Wf=fac*bigD['fit_A_model'][lag].T
    Wf-=(fac-1)*np.eye(Wf.shape[0])
    
    bigD['fit_W_matrix']=Wf.T 
    
    Ldiag,Lexc,Lzexc,Linh,Lzinh = daleMatrix_index_partition(Mt)
    print('PPF: dale partition 1st elem size: diag:%s  exc:%s  zexc:%s  inh:%s  zinh:%s'%(Mt[Ldiag[0]].shape, Mt[Lexc[0]].shape, Mt[Lzexc[0]].shape, Mt[Linh[0]].shape, Mt[Lzinh[0]].shape))
    
    Ydia=np.stack((Mt[Ldiag],Wf[Ldiag]), axis=1)
    Yexc=np.stack((Mt[Lexc],Wf[Lexc]), axis=1)
    Yinh=np.stack((Mt[Linh],Wf[Linh]), axis=1)
    Yzexc=Wf[Lzexc][Wf[Lzexc]!=0]
    Yzinh=Wf[Lzinh][Wf[Lzinh]!=0]

          
    print('diag shape: %s'%(str(Ydia.shape)))
    print('Yexc,z shape: %s %s'%(str(Yexc.shape),str(Yzexc.shape)))
    print('Yinh,z shape: %s %s'%(str(Yinh.shape),str(Yzinh.shape)))
    
    bigD['post_Ydiag']=Ydia
    bigD['post_Yexc']=Yexc
    bigD['post_Yzexc']=Yzexc
    bigD['post_Yinh']=Yinh
    bigD['post_Yzinh']=Yzinh

    pof=md['post_fit_residuals']={}
    stats,Xr,Yr = residual_stats(Ydia)
    #pprint(stats)
    pof['diag']=stats
    bigD['post_Rdia1']=Xr  # rotated 1st component
    bigD['post_Rdia2']=Yr  # rotated 2nd component

    stats,Xp,Yp = residual_stats(Yexc)
    pof['exc']=stats
    bigD['post_Rexc']=Yp
    
    stats,Xp,Yp = residual_stats(Yinh)
    pof['inh']=stats
    bigD['post_Rinh']=Yp
    #pprint(pof)

    eval_fit(md,bigD)
    
    #... construct CVS record
    outL=[md['short_name']]
    if 1:
        for obs in ['diag','exc','inh']:
            rec=pof[obs]
            x,y=rec['mu_X'],rec['mu_Y']
            if obs=='diag1' and facOn:
                r=x/(y-1)
            else:
                r=x/y
            outL+=[obs,rec['rho'],x,y,r]
            print('%s x=%.3f y=%.3f x/y=%.3f'%(obs,x,y,r))
    outL.append('')
    return outL

def nice_print_model(bigD,md,mxFeat=None):
    pmd=md['dataset']
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
    csvL=postproc_fit(expD,expMD)
  
    outF=os.path.join(args.outPath,expMD['short_name']+'.post.h5')
    write4_data_hdf5(expD,outF,expMD)

    args.prjName=expMD['short_name']
    plot=Plotter(args)  
  
    if 'a' in args.showPlots: plot.W_and_eigen(expD,expMD, tf=False,figId=1)
    if 'b' in args.showPlots: plot.Aper_row(expD,expMD, tf=False,figId=2)
    if 'c' in args.showPlots: plot.W_and_eigen(expD,expMD, tf=True,figId=1)
    if 'd' in args.showPlots: plot.Aper_row(expD,expMD, tf=True,figId=4)
    if 'e' in args.showPlots: plot.weigh_correl(expD,expMD,figId=5)
        
    plot.display_all()
    print('M:done')
    print(csvL,'\n')
    #pprint(expMD)
    #pprint(expMD['dale_truth']['5index'])
    
    pprint(expMD['fit_eval'])
   
