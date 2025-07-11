#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

'''

import os,sys

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from Util_CausalNet import print_dale_matrix
from PlotterDaleLDS import Plotter

from time import time
from pprint import pprint
import numpy as np

from time import time
import argparse
#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")         
    parser.add_argument("--basePath",default='dataDale',help="head dir for set of experimentst")
    parser.add_argument("--matrixName", default='Amats.h5', help="Path to the HDF5 file with connectivity matrices.")
      
    args = parser.parse_args()
    # make arguments  more flexible
    args.inpPath=os.path.join(args.basePath,'gen_dale')
    args.outPath=args.inpPath
    args.showPlots=''.join(args.showPlots)
      
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)

    inpF=os.path.join(args.inpPath,args.matrixName+'.daleM.h5')
    bigD,MD=read4_data_hdf5(inpF)
    pprint(MD)

    W = bigD['dale_matrix']
    eigenvalues=np.linalg.eigvals(W)
    bigD['Wtrue']=W
    bigD['Weigen']=eigenvalues

    print_dale_matrix(W,15)
    #--------------------------------
    # ....  plotting ........
    args.prjName=MD['short_name']
    #['plot']={}
    #if args.time_range!=None: expMD['plot']['time_rangeLR']=args.time_range

    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.Dale_matrix_and_eigen(bigD,MD,figId=1)
    if 'b' in args.showPlots:
        plot.Dale_stats(bigD,MD,figId=1)
   

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
