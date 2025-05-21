#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Analyze  
./postproc_ouiVar.py 

'''

import os
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from time import time
from pprint import pprint
import numpy as np
from toolbox.Util_Dale_LDS  import print_dale_matrix
from PlotterModelFit import Plotter


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a b', nargs='+',help="abcd-string listing shown plots")
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("--basePath",default='out',help="head dir for any results")
                        
    parser.add_argument('-e',"--expName",  default='fit-b7018a',help='UoI-VAR fitted model')

    parser.add_argument('-m', '--max_feature', default=20, type=int, help='max num of analyzed features')

    args = parser.parse_args()
    # make arguments  more flexible 
    
    args.modelPath=os.path.join(args.basePath,'model_uoi')
    args.outPath=os.path.join(args.basePath,'postproc')
    args.showPlots=''.join(args.showPlots)
  
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.modelPath)
    assert os.path.exists(args.outPath)
    return args


#...!...!.................... 
def nice_print_model(bigD,md,mxFeat=None):
    #pprint(md)
    pmd=md['payload']
    sem=md['selector']
    fim=md['fit_uoi']
    lag=fim['lag_depth']
    ntime,nfeat=fim['data_shape']
    inpN=sem['input_name']
    print('Postproc UoI-VAR  input=%s  nfeat=%d  ntime=%d lag=%d  fitTime=%.1f sec  ranks=%d'%(inpN,nfeat,ntime,lag,fim['fit_time'],fim['num_rank']))
    AV=bigD['fit_A_model']
   
    freqData=[i for i in range(AV[0].shape[0])]
    #1bigD['sel_feat_freq']=freqData  # tmp
    
    if mxFeat!=None:
        nfeat=min(mxFeat,nfeat)
    
    for il in range(lag):
        print('\nA_model[%d]'%(il))
        A=AV[il]
        print_dale_matrix(A,nfeat)    
 
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
                    
    inpF=args.expName+'.fitUoI.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.modelPath,inpF))
    
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        stop2
 
    if 0: # older data fix 
        pmd=expMD['payload']
        cad=expMD['canned']
       
    nice_print_model(expD,expMD,mxFeat=args.max_feature)
    
  
    #...... WRITE  OUTPUT
    outF=os.path.join(args.outPath,expMD['short_name']+'.post.h5')
    write4_data_hdf5(expD,outF,expMD)

    
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
        
    plot=Plotter(args)  
  
    if 'a' in args.showPlots:
        plot.A_matrix(expD,expMD,figId=1,lag=0)
    if 'b' in args.showPlots:
        plot.Aper_row(expD,expMD,figId=1,lag=0)

    if 'c' in args.showPlots:
        plot.A_matrix(expD,expMD,figId=1,lag=-1)
    if 'd' in args.showPlots:
        plot.Aper_row(expD,expMD,figId=1,lag=-1)
        
    if 'e' in args.showPlots:
        plot.weigh_correl(expD,expMD,figId=1)
        
    plot.display_all()
    print('M:done')
   
