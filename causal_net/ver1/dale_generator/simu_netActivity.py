#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
simu_netActivity.py

Reads a connectivity matrix from an HDF5 file and simulates the neural activity 
of the corresponding linear dynamical system (LDS). The simulation outputs state trajectory and simulation parameters are saved in an HDF5 file.

Usage:
  ./simu_netActivity.py --matrixName abc.daleM [options]

Options:
  --matrixName      Name of the HDF5 file containing connectivity matrices (without .daleM.h5 extension)
  --sigma_noise     Noise variance strength (default: 20.0)
  --binFractalNoise Use binary fractal patterns for noise modulation (default: False)
  --tau_response    (sec) Response time to driving force (default: 0.01 sec)
  -T, --evol_time   (sec) Total simulation time (default: 60 sec)
  
  --outName         Output name for simulation results (optional)
  --rnd_seed        Random seed for simulation (optional)
  --basePath        Head directory for experiments (default: dataDale)
  -v, --verb        Increase debug verbosity (default: 1)

salloc -q interactive -C cpu  -t 4:00:00 -A m2043 -N 1

"""

import os,hashlib
import sys
import argparse
import numpy as np
from pprint import pprint
from time import time

from toolbox.Util_Dale_LDS  import gen_net_activity_discrT
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5


def commandline_parser():
    parser = argparse.ArgumentParser(description="Simulate network activity from a given connectivity matrix.")
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument("--matrixName", default='Amats.h5', help="Path to the HDF5 file with connectivity matrices.")
    
    # Simulation parameters
    parser.add_argument("--sigma_noise", type=float, default=1., help="Noise variance strength.")
    parser.add_argument("--tau_response", type=float, default=10, help="(time steps) response time to driving force")
    parser.add_argument("-T","--evol_time", type=int, default=60000, help=" (time steps) Total simulation time.")

    parser.add_argument("--basePath",default='dataDale',help="head dir for set of experimentst")
    parser.add_argument("--outName", type=str, default=None, help="Output HDF5 file for simulation results.")
    parser.add_argument("--rnd_seed", type=int, default=None, help="Random seed for simulation (optional).")
  
    args = parser.parse_args()
    # make arguments  more flexible
    args.inpPath=os.path.join(args.basePath,'gen_dale')
    args.outPath=args.inpPath

    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)

    return args


#...!...!....................
def buildSimuMeta(args,md):
    dmm=md['dale_truth']
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:7]
    md['hash']=myHN

    sm={}  #  simulator
    md['simu']=sm
    
    sm['sigma_noise']=args.sigma_noise
    sm['tau_response']=args.tau_response
    sm['evol_time']=args.evol_time
    
    if args.outName!=None:
        dmm['dale_name']=md.pop('short_name')
        md['short_name']=args.outName        
    else:
        md['short_name']+='-%s'%(md['hash'])    

    
#=================================
#  M A I N 
#=================================
if __name__ == '__main__':
    args = commandline_parser()
    np.set_printoptions(precision=3)

    inpF=os.path.join(args.inpPath,args.matrixName+'.daleM.h5')
    bigD,MD=read4_data_hdf5(inpF)
    buildSimuMeta(args,MD)
    pprint(MD)   
    W = bigD.pop('dale_matrix')
    print("M:Simulating network activity M:%s  time_steps: %d  ..."%(W.shape,args.evol_time))
    
    T0=time()
    tspace,xt = gen_net_activity_discrT(W, tau=args.tau_response, sigma=args.sigma_noise, T=args.evol_time)
    print("Simulation complete, elaT=%.1f min"%((time()-T0)/60.))
    bigD['network_matrix']=W.astype(np.float32)
    bigD['evol_time']=tspace.astype(np.float32)
    bigD['evol_state']=xt.astype(np.float32)
     
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.outPath,MD['short_name']+'.simNet.h5')
    write4_data_hdf5(bigD,outF,MD)    
    print('   ./plot_simNetActivity.py  --basePath $basePath   --simName   %s -p a b   -Y    --time_range 50 1000  \n'%(MD['short_name'] ))
    print('  cd ../fit_UoI;  ./prep_simInput.py --basePath $basePath  --simName   %s   \n'%(MD['short_name'] ))
  
    exit(0)
    np.set_printoptions(precision=3, suppress=True)
    iTrial=0
    nTime=50
    t0=1000
    for i in range(5):
        iNeur=i*10
        print('\n iNeur=%d '%(iNeur))
        print('tspace:',tspace[t0:t0+nTime])
        print('xt:',xt[t0:t0+nTime,iNeur])
        print('count/h:',spike_count[iTrial,t0:t0+nTime,iNeur])
