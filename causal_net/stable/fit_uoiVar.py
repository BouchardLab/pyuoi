#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 fit UoI-VAR

Perlmutter
inside image
./fit_uoiVar.py --dataPath /m2043/DIV13/features --inpName HET_80k_1-2kHz_1ms

bare metal
./fit_uoiVar.py --dataPath /global/cfs/cdirs/m2043/causal_inference/DIV13/features --inpName HET_80k_1-2kHz_1ms --time_range 7 7.6 --lag_depth 8 --num_feature 5 

'''

import os,sys,hashlib
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from time import time
from pprint import pprint
import numpy as np

from pyuoi.linear_model import *
sys.path.append(os.path.abspath("../../"))
from examples.var_utils import *

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--basePath",default='out',help="head dir for any results")
    parser.add_argument("--dataPath",default=None,help="direct input path")
                        
    parser.add_argument("--inpName",  required=True,help='name of input data')
    parser.add_argument("--fitName",  default=None,help='fit name')

    #.... fit params
    parser.add_argument('--num_feature', default=8, type=int, help='num of from full dataset')
    parser.add_argument('--time_range' , default=[0., 1.0],  nargs=2,   type=float, help='fit data time range')
    parser.add_argument('--test_time_range' , default=None,  nargs=2,   type=float, help='test data range')
    parser.add_argument('--lag_depth', default=10, type=int, help='depth of auotorgeression')
    parser.add_argument('--max_iter', default=500, type=int, help='uoi fit limit')
    parser.add_argument('--fit_tol', default=1e-4, type=float, help='uoi fit stop condition')
    
    args = parser.parse_args()
    # make arguments  more flexible
    if args.dataPath==None:
        args.dataPath=os.path.join(args.basePath,'input')
    
    args.modelPath=os.path.join(args.basePath,'model')
   
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.modelPath)
    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    return args


#...!...!....................
def uoiVar_predict(bigD,md):
    fim=md['fit_uoi']
    data=bigD['fit_data'][:,:500]
    print('test_data:',data.shape)
    lag=fim['lag_window']
    X,Y = vectorization(data, lag)
    print('pX:',X.shape)
    print('pY:',Y.shape)
    UoI_Lasso.predict(X)
    
#...!...!....................
def fit_uoiVar(bigD,md,args):
    pmd=md['payload']
    nfeat=min(pmd['num_feature'],args.num_feature)
    maxIter=args.max_iter
    fitTol=args.fit_tol
    lag=args.lag_depth

    data=bigD['feature'][:nfeat]
    if args.time_range:  #.... clip data
        tL,tR=[int(x*pmd['sampling_freq']) for x in args.time_range ]
        print('FUV tbinLR:',tL,tR)
        assert tL < pmd['num_time_bin']
        data=data[:,tL:tR]
        
    data=data.T # to match UoI input format
    print('fit data:',data.shape,data.dtype)
        
    X,Y = vectorization(data, lag)
    print('fX:',X.shape)
    print('fY:',Y.shape)

    fim={};  md['fit_uoi']=fim
    fim['max_iter']=maxIter
    fim['fit_tol']=fitTol
    fim['lag_depth']=lag
    fim['X_shape']=list(X.shape)
    fim['data_shape']=list(data.shape)

    fim['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    if args.fitName==None:
        md['short_name']='fit-%s'%(fim['hash'])
    else:
        md['short_name']=args.fitName


    uoi_var = UoI_Lasso(n_real_features = nfeat, fit_VAR = True, max_iter=maxIter, tol=fitTol)
    t0=time()
    uoi_var.fit(X, Y)
    elaT=time()-t0
    fim['fit_time']=elaT
    model = uoi_var.coef_
    bigD['fit_model']=model
    bigD['fit_data']=data
    B_model = model.reshape((nfeat,nfeat,-1)).T
    print('B_model:',B_model.shape, 'lag;',lag)
    


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
        
    fit_uoiVar(expD,expMD,args)
    #uoiVar_predict(expD,expMD)

    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.modelPath,expMD['short_name']+'.fit.h5')
    write4_data_hdf5(expD,outF,expMD)
    pprint(expMD)
    fim=expMD['fit_uoi']
    
    print('SUM2 %s fit time %.1f sec'%(expMD['short_name'],expMD['fit_uoi']['fit_time']))
    print('SUM0,job_name,fit_time,num_feat,num_tbin,lag_depth,max_iter,fit_tol')
    print('SUM1,%s,%.1f,%d,%d,%d,%d,%1e\n'%(expMD['short_name'],fim['fit_time'],fim['data_shape'][1],fim['data_shape'][0],fim['lag_depth'],fim['max_iter'],fim['fit_tol']))

