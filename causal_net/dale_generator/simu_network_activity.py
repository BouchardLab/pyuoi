#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
simu_network_activity.py

Reads a connectivity matrix from an HDF5 file and simulates the neural activity 
of the corresponding linear dynamical system (LDS). The simulation outputs (state trajectory 
and spiking responses) and simulation parameters are saved in an HDF5 file.

Usage:
  ./simu_network_activity.py --matrixName Amats.h5 [options]

Options:
  --matrixName  Path to the HDF5 file containing connectivity matrices.
  --samp_idx    Repetition index to use (default: 0).
  --sigma       Noise variance strength (default: 1)
  --tau         Time constant for simulation (default: 3)
  --T           Total evolution time (default: 50 )
  --dt          itime step (default: 0.1 )
  --num_trials  Number of trials to simulate, shots (default: 30)
  --seed        Random seed for simulation (optional)
  --outName     Output HDF5 file name for simulation results (default: simu_activity.h5)
"""

import os,hashlib
import sys
import argparse
import numpy as np
from pprint import pprint
from time import time

# Import the utility module.
import Util_Dale_LDS as uld
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5


#### Command-line parser #####################################################
def commandline_parser():
    parser = argparse.ArgumentParser(description="Simulate network activity from a given connectivity matrix.")
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument("--matrixName", default='Amats.h5', help="Path to the HDF5 file with connectivity matrices.")
    parser.add_argument("--samp_idx", type=int, default=0, help="Dale matrix  index (default: 0).")
    
    # Simulation parameters
    parser.add_argument("--sigma_noise", type=float, default=1, help="Noise variance strength.")
    parser.add_argument("--tau_response", type=float, default=3, help="(sec) response time to driving force")
    parser.add_argument("-T","--evol_time", type=float, default=50, help=" (sec) Total simulation time.")
    parser.add_argument("-dt","--time_step", type=float, default=0.2, help=" (sec) Integration time for one evolution step")
    
    parser.add_argument("--num_trials", type=int, default=30, help="Number of shots per time step")
    parser.add_argument("--rnd_seed", type=int, default=None, help="Random seed for simulation (optional).")
    parser.add_argument("--basePath",default='dataDale',help="head dir for set of experimentst")
    parser.add_argument("--outName", type=str, default=None, help="Output HDF5 file for simulation results.")
  
    args = parser.parse_args()
    # make arguments  more flexible
    args.inpPath=args.basePath
    args.outPath=os.path.join(args.basePath,'simu')

    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)

    return args


#...!...!....................
def buildSimuMeta(args,md):
    dmm=md['dale_truth']
    dmm['hash']=md.pop('hash')
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:7]
    md['hash']=myHN

    sm={}  #  simulator
    md['simu']=sm
    
    sm['daleM_samp_idx']=args.samp_idx
    sm['sigma_noise']=args.sigma_noise
    sm['tau_response']=args.tau_response
    sm['evol_time']=args.evol_time
    sm['time_step']=args.time_step
    #sm['boxcox_lambda']=args.boxcox_lambda
    sm['num_trials']=args.num_trials
    sm['rnd_seed']=args.rnd_seed
    if args.outName!=None:
        dmm['dale_truth']=md.pop('short_name')
        md['short_name']=args.outName        
    else:
        md['short_name']+='-%s'%(md['hash'])    

    
#=================================
#  M A I N 
#=================================
if __name__ == '__main__':
    args = commandline_parser()
    np.set_printoptions(precision=3)

    inpF=os.path.join(args.inpPath,args.matrixName+'.dale.h5')
    bigD,MD=read4_data_hdf5(inpF)
    buildSimuMeta(args,MD)
    #pprint(MD)   

    # Extract the desired connectivity matrix
    rep_idx = args.samp_idx
    W = bigD.pop('dale_matrix')[rep_idx, :, :]
    
    # Run the simulation using parameters provided via command-line
    print("Simulating network activity ...")
    
    T0=time()
    tspace,xt, spike_count = uld.gen_activity(W, tau=args.tau_response, sigma=args.sigma_noise, T=args.evol_time, h=args.time_step, num_trials=args.num_trials, seed=args.rnd_seed)
    print("Simulation complete, elaT=%.1f min"%((time()-T0)/60.))
    bigD['Wdale']=W
    bigD['Time']=tspace
    bigD['Xstate']=xt
    bigD['Xcount']=spike_count
    
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.outPath,MD['short_name']+'.netActS.h5')
    write4_data_hdf5(bigD,outF,MD)    
    print('   ./plot_netActivity.py --simName   %s -p ab   -Y   \n'%(MD['short_name'] ))
  
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
