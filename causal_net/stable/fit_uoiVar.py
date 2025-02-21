#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

'''

import os
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from time import time
from pprint import pprint
import numpy as np
from PlotterFeatures import Plotter
from pyuoi.linear_model import *


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--basePath",default='out',help="head dir for set of experimentst")
                        
    parser.add_argument("--inpName",  default='exp_62a21daf',help='IBMQ experiment name assigned during submission')

    parser.add_argument('--num_feature', default=10, type=int, help='num of from full dataset')
     #parser.add_argument('--num_feature', default=10, type=int, help='num of from full dataset')
    parser.add_argument('--time_range' , default=[7., 8.],  nargs=2,   type=float, help='fit range')
    #parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    #parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")         
    
    args = parser.parse_args()
    # make arguments  more flexible 
    args.dataPath=os.path.join(args.basePath,'input')
    #1args.outPath=os.path.join(args.basePath,'model')
    #1 args.showPlots=''.join(args.showPlots)
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    #assert os.path.exists(args.outPath)
    return args


#...!...!....................
def fit_uoiVar(bigD,md):
    pmd=md['payload']
    #pom=md['postproc']
    nfeat=min(pmd['num_feature'],args.num_feature)
    #.... clip data
    tL,tR=[int(x*pmd['sampling_freq']) for x in args.time_range ]
    print('FUV tbinLR:',tL,tR)
    X=bigD['feature'][:nfeat,tL:tR]
    print('X:',X.shape)
    Y=np.zeros_like(X[0])
    maxIter=1000
    fitTol=1e-4

    uoi_var = UoI_Lasso(n_real_features = nfeat, fit_VAR = True, max_iter=maxIter, tol=fitTol)
    uoi_var.fit(X, Y)
    B_model = uoi_var.coef_
    print('B_model:',B_model)


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
                    
    inpF=args.inpName+'.act.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            print(expD)
        stop2
        
    fit_uoiVar(expD,expMD)

    exit(0)    
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
    expMD['plot']={}
    #expMD['plot']['time_rangeLR']=[0.8,1.]
    #expMD['plot']['time_rangeLR']=[7.3,7.9]

    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.input_features(expD,expMD,figId=1)

    if 'c' in args.showPlots:
        not_tested
        plot.xyz()

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
