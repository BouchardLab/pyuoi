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
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--simName", default='daleM100apr30-e7e3be2', help="[.simNet.h5]  simulated net-activation")
       
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument("--outName",  default=None,help='(optional) output file name')
 
    args = parser.parse_args()
    args.inpPath=os.path.join(args.basePath,'gen_dale')
    args.outPath=os.path.join(args.basePath,'input_uoi')
    
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    
    return args

#...!...!....................
def format_simNetActivity(inpD,inpMD):

    # prep meta-data
    smd=inpMD['simu']         
    sem={}
    md={'selector':sem, 'payload':smd}

    sem['input_name']=args.simName
    sem['sampling_freq'] =1./smd['time_step']
        
    md['hash']=inpMD['hash']
    if args.outName!=None:
        md['short_name']=args.outName
    else:
        md['short_name']=inpMD['short_name']
        
    stateV=inpD['evol_state']
    #.... any data transformation goes here ....
    outD={'features':stateV,'true_network_matrix':inpD['network_matrix']}
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

    expMD,expD=format_simNetActivity(inpD,inpMD)
    
    pprint(expMD)
   
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.act.h5')
    write4_data_hdf5(expD,outF,expMD)
    #1print('   ./plot_features.py  --basePath $basePath   --inpName   %s  -p  a c d  -Y '%(expMD['short_name'] ))
    print('   ./fit_uoiVar_admm.py  --basePath $basePath   --inpName   %s    --time_range 0.3 1.3  \n'%(expMD['short_name'] ))
   

    print('1 node: \n     srun -n128 --distribution=block:block shifter python  fit_uoiVar_admm.py  --basePath $basePath   --inpName   %s    --time_range 0.3 1.3  \n'%(expMD['short_name'] ))
    
