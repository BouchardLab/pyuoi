#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

from Util_NumpyIO import read_data_npz


def main():
    parser = argparse.ArgumentParser(
        description="Print shape/size/dtype for A_debias and B_debias in a prismEM npz."
    )
    parser.add_argument(
        "inpF",
        nargs="?",
        default="/pscratch/sd/b/balewski/2026_causalNet_exp_ver3c/prismFit/daleN200_74e6d6_e2b7d7_take1_10minC.prismEM.npz",
        help="input npz file",
    )
    args = parser.parse_args()

    inpF = Path(args.inpF)
    dataD, _ = read_data_npz(inpF, verb=0)

    for name in ("A_debias", "B_debias"):
        if name not in dataD:
            print(f"missing key: {name}", file=sys.stderr)
            continue
        arr = dataD[name]
        print(f"{name}: shape={arr.shape}, size={arr.size}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
