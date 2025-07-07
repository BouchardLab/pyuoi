#!/usr/bin/env python3
import os
import argparse
import numpy as np

def inspect_spike_times(fname, max_keys=5, max_vals=6):
    """
    Load spike_times.npy and print its structure.
     - If it’s a 0-d object array, unwrap via .item()
     - If it’s a dict, show first `max_keys` keys and up to `max_vals` values each
     - Otherwise handle list/tuple or numeric arrays
    """
    print('Opening:',fname,flush=True)
    raw = np.load(fname, allow_pickle=True)
    print(f"Loaded '{fname}' → type={type(raw)}, dtype={getattr(raw,'dtype',None)}, shape={getattr(raw,'shape',None)}")

    # unwrap zero‐dim object‐array
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        data = raw.item()
        print("Unwrapped 0-d object array; now data is", type(data))
    else:
        data = raw

    # Case A: dict
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"\nDetected dict with {len(keys)} keys.")
        print('Sample keys:',keys[:20],'...', keys[-20:])
        for i, k in enumerate(keys[:max_keys]):
            v = data[k]
            print(f"\nKey [{i}] = {k!r}:  type={type(v)}")
            # try to view as array
            try:
                arr = np.asarray(v)
                # flatten and take first max_vals elements
                flat = arr.ravel()
                print(f"  shape={arr.shape}, dtype={arr.dtype}")
                if flat.size>0:
                    vals = flat[:max_vals]
                    str_vals = [f"{v:.6f}" for v in vals]
                    print(f"  first {len(str_vals)} values = [{', '.join(str_vals)}]")
                    
                else:
                    print("  (empty array)")
            except Exception:
                print("  (could not convert to array, repr:)", repr(v))
        if len(keys)>max_keys:
            print(f"\n  … and {len(keys)-max_keys} more keys not shown.")

    # Case B: list/tuple or object-dtype array
    elif isinstance(data, (list, tuple)) or (isinstance(data, np.ndarray) and data.dtype==object):
        Nn = len(data)
        print(f"\nDetected sequence of length {Nn}; assuming per-neuron spike times.")
        for i, times in enumerate(data[:max_keys]):
            try:
                arr = np.asarray(times, dtype=float)
                flat = arr.ravel()
                print(f"  neuron[{i}]: {flat.size} spikes", end='')
                if flat.size>0:
                    snippet = flat[:max_vals]
                    print(f", first {len(snippet)} = {snippet.tolist()}")
                else:
                    print()
            except Exception:
                print(f"  neuron[{i}]: could not convert entry to array, repr={repr(times)}")
        if Nn>max_keys:
            print(f"  … and {Nn-max_keys} more neurons not shown.")

    # Case C: plain numeric array
    else:
        arr = np.asarray(data)
        print(f"\nDetected numeric array of shape {arr.shape}, dtype={arr.dtype}")
        if arr.size>0 and arr.dtype.kind in ('i','f'):
            flat = arr.ravel()
            print("  min=", flat.min(), " max=", flat.max(),
                  " mean={:.3g}".format(flat.mean()), " std={:.3g}".format(flat.std()))
            if set(np.unique(flat)) <= {0,1}:
                print("  (0/1 array) total 1s =", int(flat.sum()))
        else:
            print("  (empty or non-numeric array)")

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataPath", "-d",
                   help="Directory containing spike_times.npy",
                   default="/global/cfs/cdirs/m2043/causal_inference/Canine_Organoids_PVS/Analysis/250619/M08020/Network/")
    p.add_argument("--sessionName",  default='000093/well000',help='raw data session name')
    args = p.parse_args()
    
     
    data_file = os.path.join(args.dataPath, args.sessionName,"spike_times.npy")
    if not os.path.isfile(data_file):
        print(f"Error: file not found: {data_file}")
        return

    inspect_spike_times(data_file)

if __name__ == "__main__":
    main()
