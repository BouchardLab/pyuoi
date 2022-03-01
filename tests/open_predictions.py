import argparse
import numpy as np
from numpy.lib.npyio import NpzFile
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


def graph_2d_subset_linear_classification_coefficients(key: str, data: NpzFile, frame_start: int = 0, frame_end: int = 10) -> None:
    """
    Print out a subset of frames from the corresponding key in the data (.npz) file.
    """
    with np.printoptions(precision=2, suppress=True):
        fig, ax = plt.subplots()
        ax.set_title(f"{key} frames from: {frame_start} - {frame_end}")
        ax.set_xlabel("frames")
        ax.set_ylabel("n_bootstraps")
        coef = np.squeeze(data[key])
        plot = ax.imshow(coef[:24, frame_start:frame_end], interpolation='none',
                         extent=[0, 10, 24, 0])
        plt.figtext(0, 0, str(coef[:24, frame_start:frame_end]), fontsize=6)
        fig.colorbar(plot, ax=ax)
        plt.show()
        plt.close()


def main(file: str):
    """Prints the first and last 10 results from the file and generates a graph of results"""

    if file.endswith(".npy"):
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
    elif file.endswith(".npz"):
        with np.load(file) as data:
            for key in data.keys():
                if key == "x_coefficients":
                    graph_2d_subset_linear_classification_coefficients(
                        key="x_coefficients", data=data, frame_start=70000, frame_end=70010)
                elif key == "y_coefficients":
                    graph_2d_subset_linear_classification_coefficients(
                        key="y_coefficients", data=data, frame_start=70000, frame_end=70010)


if __name__ == "__main__":
    arg_parser = initialize_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.inputFile):
        main(parsed_args.inputFile)
