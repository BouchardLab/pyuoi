#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
prepares input for UoI fit simulated Dale-generator data

HD5 arrays contain input and output

Use case:

*** on PM
basePath=/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp/

./prep_simInput.py --sessionName $ses --outName $ses2  --basePath $basePath
./plot_features.py   --basePath $basePath --inpName   $ses2 -p  d -Y


'''
import sys,os,hashlib
import numpy as np
import pickle
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

import argparse

#...!...!..................
def save_npy_arrays(stateV, networkM, save_path, name_prefix):
    """Save state vector and network matrix as NPY files in float16 format.
    
    Args:
        stateV: state vector array
        networkM: network matrix array
        save_path: directory to save the files
        name_prefix: prefix for the file names
    """
    xt_path = os.path.join(save_path, f"Xt_{name_prefix}.npy")
    a_path = os.path.join(save_path, f"A_{name_prefix}.npy")
    
    np.save(xt_path, stateV.astype(np.float16))
    np.save(a_path, networkM.astype(np.float16))
    
    print(f'Saved {xt_path} shape:{stateV.shape} dtype:float16')
    print(f'Saved {a_path} shape:{networkM.shape} dtype:float16')

#...!...!..................
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--basePath",default='out',help="head dir for set of experimentst")
    parser.add_argument("--simName",  required=True,help='name of input data')
    parser.add_argument("--outName",  default=None,help='output name')
    parser.add_argument("--time_start", type=float, default=0.,help="start time (sec)")
    parser.add_argument("--saveNPY", type=str, default=None, help="save state and network matrices as NPY files with this name prefix")
    
    args = parser.parse_args()
    args.inpPath=os.path.join(args.basePath,'gen_dale')
    args.outPath=os.path.join(args.basePath,'input_uoi')
    
    if args.verb>0:
        print( 'myArg-program:',parser.prog)
        for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
        print('',flush=True)
    
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def format_simNetActivity(inpD,inpMD):
    
    # prep meta-data
    smd=inpMD['simu']         
    sem={}
    md={'selector':sem, 'payload':smd}
    #pprint(inpMD['dale_truth']); aa
    md['dale_truth']=inpMD['dale_truth']
    
    sem['input_name']=args.simName
    fr=sem['sampling_freq'] =1./smd['time_step']
    sem['time_start']=args.time_start
    
    md['hash']=inpMD['hash']
    if args.outName!=None:
        md['short_name']=args.outName
    else:
        md['short_name']=inpMD['short_name']
        
    stateV=inpD['evol_state']
    #.... clip data
    tL=int(args.time_start*fr)
    assert tL < stateV.shape[0]
    stateV=stateV[tL:]
    
    #.... any data transformation goes here ....
    outD={'all_features':stateV.astype(np.float16),'true_network_matrix':inpD['network_matrix'].astype(np.float16)}

    # Compute true_matrix_5index for Dale matrix partitioning
    Mt = inpD['network_matrix'].T
    from toolbox.Util_CausalNet import daleMatrix_index_partition
    Ldia, Lexc, Lzexc, Linh, Lzinh = daleMatrix_index_partition(Mt)
    md['dale_truth']['5index'] = {
        'diag': Mt[Ldia[0]].shape[0],
        'exc_nonzero': Mt[Lexc[0]].shape[0],
        'exc_zero': Mt[Lzexc[0]].shape[0],
        'inh_nonzero': Mt[Linh[0]].shape[0],
        'inh_zero': Mt[Lzinh[0]].shape[0]
    }

    # Save NPY files if requested
    if args.saveNPY is not None:
        save_npy_arrays(stateV, inpD['network_matrix'], args.outPath, args.saveNPY)

    return md,outD
    
 
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    args=commandline_parser()
    np.set_printoptions(precision=4)
    
    inpF=os.path.join(args.inpPath,args.simName+'.simNet.h5')
    inpD,inpMD=read4_data_hdf5(inpF)

    if 1: # patch for old data
        if 'dale_truth' in inpMD['dale_truth']: inpMD['dale_truth']['dale_name']=inpMD['dale_truth'].pop('dale_truth')

    expMD,expD=format_simNetActivity(inpD,inpMD)
    
    pprint(expMD)
   
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.act.h5')
    write4_data_hdf5(expD,outF,expMD)
    #1print('   ./plot_features.py  --basePath $basePath   --inpName   %s  -p  a c d  -Y '%(expMD['short_name'] ))
    print('   ./fit_uoiVar_admm.py  --basePath $basePath   --inpName   %s    --time_range 0. 2.  \n'%(expMD['short_name'] ))
   
    print('1 node: \n     srun -n128 --distribution=block:block shifter python  fit_uoiVar_admm.py  --basePath $basePath   --inpName   %s  --num_admm 32  --time_range 0. 4.  \n'%(expMD['short_name'] ))
    
    #pprint(expMD)
