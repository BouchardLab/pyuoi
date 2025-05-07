#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 plot input features

'''

import os,sys
#import pickle
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

# tmp:
sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/causal_net/fit_UoI")
from  postproc_ouiVar  import print_Amatrix


from time import time
from pprint import pprint
import numpy as np
from Plotter_Dale_LDS import Plotter
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
    parser.add_argument("--rep", type=int, default=0, help="Repetition index (default: 0).")
     
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

#...!...!....................
def compute_eigenvalues(A):
    """
    Compute all eigenvalues of a given 2D numpy array and measure the time taken.
    
    Parameters:
      A (np.ndarray): A square matrix.
    
    Returns:
      eigenvalues (np.ndarray): Array of eigenvalues.
      comp_time (float): Time taken for the computation (in seconds).
    """
    start_time = time()
    eigenvalues = np.linalg.eigvals(A)
    comp_time = time() - start_time
    print(f"Time for computing eigenvalues: {comp_time:.1f} seconds")
    return eigenvalues, comp_time



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

    # Extract the desired connectivity matrix
    rep_idx = args.rep
    W = bigD['dale_matrix'][rep_idx, :, :]
    eigenvalues, comp_time=compute_eigenvalues(W)
    bigD['Wtrue']=W
    bigD['Weigen']=eigenvalues

    print_Amatrix(W,15)
    #--------------------------------
    # ....  plotting ........
    args.prjName=MD['short_name']
    #['plot']={}
    #if args.time_range!=None: expMD['plot']['time_rangeLR']=args.time_range

    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.Dale_matrix(bigD,MD,figId=1)
    if 'b' in args.showPlots:
        plot.Dale_eigen(bigD,MD,figId=1)
    if 'c' in args.showPlots:
        plot.Dale_stats(bigD,MD,figId=1)
   

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp
