import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def initialize_arg_parser():
    parser = argparse.ArgumentParser(
        description='Read in numpy files and visualize frequency differences')
    parser.add_argument('--inputFile',
                        help='Path to the input file.',
                        default='/Users/josephgmaa/pyuoi/pyuoi/data/features/saved_runs/20220208-154256.nolj_Recording_day7_overnight_636674151185633714_5_nolj.c3d.1851.features.netcdf.npy')
    return parser


def main(file: str):
    """Prints the first and last 10 results from the file and generates a graph of results"""

    with np.printoptions(edgeitems=10):
        results = np.load(
            file=file, allow_pickle=True)

        x, y = np.unique(results, return_counts=True)

        plt.bar(x, y)
        plt.title(os.path.basename(file))
        plt.show()
        plt.close()

        plt.scatter(np.arange(len(results)), results, s=0.3)
        plt.show()
        plt.close()


if __name__ == "__main__":
    arg_parser = initialize_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.inputFile):
        main(parsed_args.inputFile)
