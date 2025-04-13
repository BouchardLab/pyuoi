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
  --rep         Repetition index to use (default: 0).
  --sigma       Noise variance strength (default: 1)
  --tau         Time constant for simulation (default: 3)
  --T           Total simulation time (default: 1000)
  --h           Integration time resolution (default: 1e-1)
  --boxcox      Box-Cox transformation parameter (default: 0.5; use 'None' for raw counts)
  --num_trials  Number of trials to simulate (default: 10)
  --seed        Random seed for simulation (optional)
  --outName     Output HDF5 file name for simulation results (default: simu_activity.h5)
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from pprint import pprint

# Import the utility module.
import Util_Dale_LDS as uld

#### Command-line parser #####################################################
def commandline_parser():
    parser = argparse.ArgumentParser(description="Simulate network activity from a given connectivity matrix.")
    parser.add_argument("--matrixName", default='Amats.h5', help="Path to the HDF5 file with connectivity matrices.")
    parser.add_argument("--rep", type=int, default=0, help="Repetition index (default: 0).")
    
    # Simulation parameters
    parser.add_argument("--sigma", type=float, default=1, help="Noise variance strength.")
    parser.add_argument("--tau", type=float, default=3, help="Time constant for simulation.")
    parser.add_argument("--T", type=float, default=1000, help="Total simulation time.")
    parser.add_argument("--h", type=float, default=1e-1, help="Integration time resolution.")
    parser.add_argument("--boxcox", type=lambda x: None if x=="None" else float(x), default=0.5,
                        help="Box-Cox transformation parameter (use 'None' for raw counts).")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of simulation trials.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for simulation (optional).")
    parser.add_argument("--outName", type=str, default="simu_activity.h5", help="Output HDF5 file for simulation results.")
    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    return args

#### Main ###############################################################
if __name__ == '__main__':
    args = commandline_parser()
    
    print("Loading connectivity matrices from:", args.matrixName)
    with h5py.File(args.matrixName, "r") as h5f:
        matrices = h5f["matrices"][:]
        meta = dict(h5f.attrs)
    pprint(meta)
    
    # Extract the desired connectivity matrix
    rep_idx = args.rep
    
    try:
        W = matrices[rep_idx, :, :]
    except IndexError as e:
        print("Error: Invalid rep or r_index. Details:", e)
        sys.exit(1)
    print(f"Selected matrix from rep {rep_idx} (shape: {W.shape}).")
    
    # Run the simulation using parameters provided via command-line
    print("Simulating network activity ...")
    tspace,xt, spike_rates_trials = uld.gen_activity(W, tau=args.tau, sigma=args.sigma, T=args.T,
                                              h=args.h, boxcox=args.boxcox, num_trials=args.num_trials,
                                              seed=args.seed)
    print("Simulation complete.")
    
    # Save simulation output and parameters to HDF5
    with h5py.File(args.outName, "w") as h5f:
        h5f.create_dataset("tspace", data=tspace)
        h5f.create_dataset("xt", data=xt)
        h5f.create_dataset("spike_rates_trials", data=spike_rates_trials)
        # Save simulation parameters as attributes
        h5f.attrs["tau"] = args.tau
        h5f.attrs["sigma"] = args.sigma
        h5f.attrs["T"] = args.T
        h5f.attrs["h"] = args.h
        h5f.attrs["boxcox"] = args.boxcox if args.boxcox is not None else -1  # using -1 to denote None
        h5f.attrs["num_trials"] = args.num_trials
        h5f.attrs["selected_rep"] = rep_idx
        
    print(f"Saved simulation output to '{args.outName}'.")

    np.set_printoptions(precision=3, suppress=True)
    iTrial=0
    nTime=50
    t0=5000
    for i in range(5):
        iNeur=i*10
        print('\n iNeur=%d '%(iNeur))
        print('tspace:',tspace[t0:t0+nTime])
        print('xt:',xt[t0:t0+nTime,iNeur])
        print('rate:',spike_rates_trials[iTrial,t0:t0+nTime,iNeur])
