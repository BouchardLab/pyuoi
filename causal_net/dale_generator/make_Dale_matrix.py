#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
make_Dale_matrix.py

Creates Dale-type connectivity matrices with specified parameters and saves both
the matrices and metadata in an HDF5 file.

Usage:
  ./make_Dale_matrix.py [options]

Options:
  --reps      Number of matrix repetitions (default: 1)
  --M         Number of excitatory neurons (default: 30)
  --p         Synaptic connection probability (default: 0.25)
  --g         Inhibitory scaling factor (default: 2)
  --R_value   scaling of network activity
  --outName   Output HDF5 file name (default: Amats.h5)
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm

# Import the utility module without global variables.
import Util_Dale_LDS as uld

#### Command-line parser #####################################################
def commandline_parser():
    parser = argparse.ArgumentParser(description="Generate Dale LDS connectivity matrices.")
    parser.add_argument("--reps", type=int, default=1, help="Number of matrix repetitions.")
    parser.add_argument("--M", type=int, default=30, help="Number of excitatory neurons.")
    parser.add_argument("--p", type=float, default=0.25, help="Synaptic connection probability.")
    parser.add_argument("--g", type=float, default=2, help="Inhibitory scaling factor (gamma).")
    parser.add_argument("-R", "--R_value", type=float, default=1.3, help="scaling of network activity")
    parser.add_argument("--outName", type=str, default="Amats.h5", help="Output HDF5 file name.")
    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    return args

#### Main ###############################################################
if __name__ == '__main__':
    args = commandline_parser()

    print("Generating matrices with  R value:  %.3f"%args.R_value)
    
    # Generate the connectivity matrices (no temporary pickle file is used)
    Alist = uld.gen_matrices(args.reps, args.M, args.p, args.g, args.R_value)
    print("Matrix generation complete.")
    
    # Convert the nested list of matrices to a NumPy array.
    # Expected shape: (reps, num_R, 2*M, 2*M)
    A_stack = np.array(Alist)
    print("Converted matrices to numpy array with shape:", A_stack.shape)
    
    # Save matrices and metadata to HDF5
    with h5py.File(args.outName, "w") as h5f:
        h5f.create_dataset("matrices", data=A_stack)
        # Store metadata as attributes
        h5f.attrs["reps"] = args.reps
        h5f.attrs["M"] = args.M
        h5f.attrs["p"] = args.p
        h5f.attrs["g"] = args.g
        h5f.attrs["R_scale"] = args.R_value
       
    
    print(f"Saved connectivity matrices and metadata to '{args.outName}'.")
