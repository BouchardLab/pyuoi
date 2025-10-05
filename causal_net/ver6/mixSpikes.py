#!/usr/bin/env python3
"""
Neuron mixing tool for simulated Dale Poisson network data.

This script modifies simulated neural network data by selectively removing
neurons according to various strategies. It processes both connectivity
matrices and spike data to create modified datasets for testing robustness
and generalization of connectivity inference methods.

Main functionality includes:
- Random neuron removal (dropAny): Randomly removes specified fraction of neurons
- Low-frequency neuron removal (dropLowFreq): Removes neurons with lowest firing rates
- Preserves data structure and updates index mappings
- Creates new datasets with modified metadata

Usage:
    ./mixSpikes.py --inpSimName daleM150_448b86 --action dropAny 0.2 --mixTag mix1
    ./mixSpikes.py --inpSimName daleM150_448b86 --action dropLowFreq 0.3 --mixTag lowFreqDrop
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
import numpy as np
import argparse
from pprint import pprint
from toolbox.Util_NumpyIO import read_data_npz, write_data_npz

#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser(description="Mix/modify simulated Dale Poisson spike data by removing neurons")
    parser.add_argument("-v","--verbosity",type=int, help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--dataPath",default='/pscratch/sd/b/balewski/2025_causalNet_tmp/',help="head dir for input/output data")
    parser.add_argument("--inpSimName", type=str, required=True, help="Input simulation base name (output from sim_dalePoisson.py)")
    parser.add_argument("--action", nargs='+', required=True, help="Action to perform: dropAny X or dropLowFreq X where X is fraction (e.g., dropAny 0.2 for 20%%)")
    parser.add_argument("--mixTag", type=str, required=True, help="Tag appended to output name (e.g., 'mix1')")
    
    args = parser.parse_args()
    
    print('myArg-program:',parser.prog)
    for arg in vars(args):  print('myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    return args

#...!...!....................
def parse_action(action_list):
    if len(action_list) != 2:
        raise ValueError(f"Action must be 'dropAny X' or 'dropLowFreq X', got: {action_list}")
    action_type, fraction_str = action_list
    try:
        fraction = float(fraction_str)
    except ValueError:
        raise ValueError(f"Fraction must be a number, got: {fraction_str}")
    if not 0 < fraction < 1:
        raise ValueError(f"Fraction must be between 0 and 1, got: {fraction}")
    return action_type, fraction

#...!...!....................
def select_neurons_to_keep(action_type, fraction, Nn, single_rates, num_excite):
    n_drop = int(Nn * fraction)
    n_keep = Nn - n_drop
    print(f"\nSelecting neurons to keep: action={action_type}, fraction={fraction:.2f}")
    print(f"  Original neurons: {Nn} (excit={num_excite}, inhib={Nn-num_excite}), dropping: {n_drop}, keeping: {n_keep}")
    
    if action_type == 'dropAny':
        keep_indices = np.sort(np.random.choice(Nn, size=n_keep, replace=False))
        print(f"  Random selection: kept neuron indices range [{keep_indices[0]}, {keep_indices[-1]}]")
    elif action_type == 'dropLowFreq':
        freq_sorted_indices = np.argsort(single_rates)
        keep_indices = np.sort(freq_sorted_indices[n_drop:])
        dropped_freq_range = [single_rates[freq_sorted_indices[0]], single_rates[freq_sorted_indices[n_drop-1]]]
        kept_freq_range = [single_rates[keep_indices[0]], single_rates[keep_indices[-1]]]
        print(f"  Dropped frequency range: [{dropped_freq_range[0]:.2f}, {dropped_freq_range[1]:.2f}] Hz")
        print(f"  Kept frequency range: [{kept_freq_range[0]:.2f}, {kept_freq_range[1]:.2f}] Hz")
    else:
        raise ValueError(f"Unknown action type: {action_type}. Must be 'dropAny' or 'dropLowFreq'")
    
    num_excite_kept = np.sum(keep_indices < num_excite)
    print(f"  Kept neurons: excit={num_excite_kept}, inhib={n_keep-num_excite_kept}")
    
    return keep_indices, num_excite_kept

#...!...!....................
def remove_neurons(trueD, spikeD, keep_indices):
    print("\n=== Removing neurons from data ===")
    Nn_orig = trueD['A_true'].shape[0]
    Nn_new = len(keep_indices)
    
    print(f"Original size: {Nn_orig} neurons")
    print(f"New size: {Nn_new} neurons")
    
    # Remove from connectivity matrix (both rows and columns)
    A_new = trueD['A_true'][np.ix_(keep_indices, keep_indices)]
    print(f"  A_true: {trueD['A_true'].shape} -> {A_new.shape}")
    
    # Remove from bias vector
    B_new = trueD['B_true'][keep_indices]
    print(f"  B_true: {trueD['B_true'].shape} -> {B_new.shape}")
    
    # Remove from spike data (all time points, selected neurons)
    spikes_new = spikeD['spikes'][:, keep_indices]
    print(f"  spikes: {spikeD['spikes'].shape} -> {spikes_new.shape}")
    
    # Remove from firing rates
    rates_new = spikeD['single_rates'][keep_indices]
    print(f"  single_rates: {spikeD['single_rates'].shape} -> {rates_new.shape}")
    
    # Update index mappings - create new sequential indices
    neur_revFreqIdx_new = np.arange(Nn_new)
    neur_freqIdx_new = np.arange(Nn_new)
    print(f"  Index mappings recreated for {Nn_new} neurons")
    
    # Create output dictionaries
    trueD_new = {
        'A_true': A_new,
        'B_true': B_new,
        'neur_freqIdx': neur_freqIdx_new,
        'neur_revFreqIdx': neur_revFreqIdx_new
    }
    
    spikeD_new = {
        'spikes': spikes_new,
        'single_rates': rates_new
    }
    
    
    return trueD_new, spikeD_new

#...!...!....................
def update_metadata(trueMD, spikeMD, args, action_type, fraction, Nn_orig, Nn_new, num_excite_new):
    print("\n=== Updating metadata ===")
   
    # Update simTruth metadata
    trueMD_new = {}
    trueMD_new['short_name'] = f"{args.inpSimName}-{args.mixTag}"
    if 'evol_conf' in trueMD:
        trueMD_new['evol_conf'] = trueMD['evol_conf']
    if 'dale_conf' in trueMD:
        dale_conf_new = trueMD['dale_conf'].copy()
        dale_conf_new['num_neurons'] = Nn_new
        dale_conf_new['num_excite'] = int(num_excite_new)
        trueMD_new['dale_conf'] = dale_conf_new
        print(f"  simTruth metadata: updated 'dale_conf' with num_neurons={Nn_new}, num_excite={num_excite_new}")
    trueMD_new['input_mixer'] = {
        'inpSimName': args.inpSimName,
        'action': ' '.join(args.action),
        'mixTag': args.mixTag,
        'num_neurons_orig': Nn_orig,
        'num_neurons_new': Nn_new,
        'dropped_fraction': fraction,
        'dropped_count': Nn_orig - Nn_new
    }
    
    print(f"  simTruth metadata: removed 'dale_simu_stats'")
    print(f"  simTruth metadata: updated 'short_name' to {trueMD_new['short_name']}")
    print(f"  simTruth metadata: created 'input_mixer' with action info")
    
    # Update spike metadata
    spikeMD_new = spikeMD.copy()
    spikeMD_new['short_name'] = f"{args.inpSimName}-{args.mixTag}"
    print(f"  spike metadata: updated 'short_name' to {spikeMD_new['short_name']}")
    #pprint(trueMD_new); aaa
    return trueMD_new, spikeMD_new

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args = get_parser()
    np.set_printoptions(precision=3)
    
    # Parse action list
    action_type, fraction = parse_action(args.action)
    
    # Load input simulation data
    print(f"\n=== Loading input data: {args.inpSimName} ===")
    truthFF = os.path.join(args.dataPath, f"{args.inpSimName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb>0)
    
    spikesFF = os.path.join(args.dataPath, f"{args.inpSimName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb>0)
    
    if args.verb > 1:
        print("\nOriginal simTruth metadata:")
        pprint(trueMD)
        print("\nOriginal spike metadata:")
        pprint(spikeMD)
    
    # Get original neuron counts
    Nn_orig = trueD['A_true'].shape[0]
    num_excite_orig = trueMD.get('dale_conf', {}).get('num_excite', Nn_orig)
    
    # Select neurons to keep
    keep_indices, num_excite_new = select_neurons_to_keep(action_type, fraction, Nn_orig, spikeD['single_rates'], num_excite_orig)
    
    # Remove neurons from data
    trueD_new, spikeD_new = remove_neurons(trueD, spikeD, keep_indices)
    
    # Update metadata
    Nn_new = len(keep_indices)
    trueMD_new, spikeMD_new = update_metadata(trueMD, spikeMD, args, action_type, fraction, Nn_orig, Nn_new, num_excite_new)
    
    # Write output files
    outName = f"{args.inpSimName}-{args.mixTag}"
    print(f"\n=== Writing output files: {outName} ===")
    
    outTruthFF = os.path.join(args.dataPath, f"{outName}.simTruth.npz")
    write_data_npz(trueD_new, outTruthFF, metaD=trueMD_new)
    print(f"  Wrote: {outTruthFF}")
    
    outSpikesFF = os.path.join(args.dataPath, f"{outName}.spikes.npz")
    write_data_npz(spikeD_new, outSpikesFF, metaD=spikeMD_new)
    print(f"  Wrote: {outSpikesFF}")
    
    if args.verb > 1:
        print("\nNew simTruth metadata:")
        pprint(trueMD_new)
        print("\nNew spike metadata:")
        pprint(spikeMD_new)
    
    print("\nMixing completed successfully!")
    print("\nNext step commands:")
    print(f"  ./view_dalePoisson.py  --dataName {outName}  -p a b c")
    print(f"  ./fit_lassoPoisson.py  --dataName {outName}  --num_epochs 50")
    print("    --dataPath "+args.dataPath)
    print("M:done")

