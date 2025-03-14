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

import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")         
    parser.add_argument("--basePath",default='out',help="head dir for set of experimentst")
    parser.add_argument('--time_range' , default=[0., 1.0],  nargs=2,   type=float, help='fit data time range')
    parser.add_argument("--inpName",  default='exp_62a21daf',help='IBMQ experiment name assigned during submission')
    
    args = parser.parse_args()
    # make arguments  more flexible 
    args.dataPath=os.path.join(args.basePath,'features')
    args.outPath=os.path.join(args.basePath,'post')
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
                    
    inpF=args.inpName+'.act.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            print(expD)
        stop2
        
      
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
    expMD['plot']={}
    #expMD['plot']['time_rangeLR']=[0.,10.]
    expMD['plot']['time_rangeLR']=[7.4,7.85]
    #expMD['plot']['time_rangeLR']=[0,10]
    #if args.time_range!=None: expMD['plot']['time_rangeLR']=args.time_range

    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.input_features(expD,expMD,figId=1)
    if 'b' in args.showPlots:
        plot.input_features_dense(expD,expMD,figId=2)

    if 'c' in args.showPlots:
        plot.global_qa(expD,expMD,figId=2)
        
    if 'd' in args.showPlots:
        plot.detailed_qa(expD,expMD,figId=2)

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
