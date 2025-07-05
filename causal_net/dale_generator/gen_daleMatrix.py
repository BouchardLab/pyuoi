#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
make_Dale_matrix.py

Creates Dale-type connectivity matrices with specified parameters and saves both
the matrices and metadata in an HDF5 file.

Usage:
  ./make_daleMatrix.py [options]

Options:
  --ne        Number of excitatory neurons 
  --p         Synaptic connection probability 
  --g         Inhibitory scaling factor 
  --activity_scale   scaling of network activity
  --outName   Output HDF5 file name (default: daleM-537645v.daleM.h5)
"""

import sys,os,hashlib
import argparse
import numpy as np
from pprint import pprint
from time import time

from toolbox.Util_Dale_LDS  import gen_matrices
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

def commandline_parser():
    parser = argparse.ArgumentParser(description="Generate Dale LDS connectivity matrices.")
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument("--num_samp", type=int, default=1, help="Number of matrix instantiations")
    parser.add_argument("-ne","--num_excit_neur", type=int, default=50, help="Number of excitatory neurons.")
    parser.add_argument("-p","--prob_synaptic_conn", type=float, default=0.1, help="Synaptic connection probability.")
    parser.add_argument("-g","--gamma_inhib", type=float, default=2., help="Inhibitory scaling factor (gamma).")
    parser.add_argument("-R", "--activity_scale", type=float, default=2., help="scaling of network activity")
    parser.add_argument("--matrixName", type=str, default=None, help=" [.h5] Output HDF5 file name.")
    parser.add_argument("--basePath",default='dataDale',help="head dir for set of experiments")
    
    args = parser.parse_args()
    args.outPath=os.path.join(args.basePath,'gen_dale')
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def buildDaleMeta(args):
    dmm={}  #  dale-matrix
    dmm['num_excit_neur']=args.num_excit_neur
    dmm['prob_synaptic_conn']=args.prob_synaptic_conn
    dmm['gamma_inhib']=args.gamma_inhib
    dmm['activity_scale']=args.activity_scale
    dmm['num_DaleM_samp']=args.num_samp
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:7]
    dmm['hash']=myHN
    md={ 'dale_truth':dmm}
    if args.matrixName==None:
        md['short_name']='daleM%d-%s'%(2*dmm['num_excit_neur'],dmm['hash'])
    else:
        md['short_name']=args.matrixName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md


#=================================
#  M A I N 
#=================================
if __name__ == '__main__':
    args = commandline_parser()
    MD=buildDaleMeta(args)
        
    print("Generating %d Dale matrices.."%args.num_samp)
    T0=time()
    A_stack = gen_matrices( args.num_excit_neur, args.prob_synaptic_conn, args.gamma_inhib, args.activity_scale)
    print("Matrix generation complete, elaT=%.1f min"%((time()-T0)/60.))
    
    print("Matrices  array with shape:", A_stack.shape)
    MD['dale_truth']['num_any_neur']=A_stack.shape[1]
    pprint(MD)
    if A_stack.shape[0]==1 : # alwasy the case
        A_stack=A_stack[0] # drop axis=0
    bigD={'dale_matrix':A_stack}

    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.outPath,MD['short_name']+'.daleM.h5')
    write4_data_hdf5(bigD,outF,MD)    
    print('   ./plot_daleMatrix.py  --basePath $basePath  --matrixName   %s -p abc   -Y   '%(MD['short_name'] ))
    print('   ./simu_netActivity.py  --basePath $basePath  --matrixName   %s   \n'%(MD['short_name'] ))
   
