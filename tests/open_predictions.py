import argparse
import numpy as np
from numpy.lib.npyio import NpzFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import json
import base64


def initialize_arg_parser():
    parser = argparse.ArgumentParser(
        description='Read in numpy files and visualize frequency differences')
    parser.add_argument('--input_file',
                        help='Path to the input file.',
                        default='/Users/josephgmaa/pyuoi/pyuoi/data/features/saved_runs/20220208-154256.nolj_Recording_day7_overnight_636674151185633714_5_nolj.c3d.1851.features.netcdf.npy')
    parser.add_argument('--key',
                        help='Key for reading JSON keys',
                        default='coef_')
    return parser


def graph_2d_subset_x_linear_classification_coefficients(key: str, data: NpzFile, frame_start: int = 0, frame_end: int = 10, feature_idx: int = 0) -> None:
    """
    Print out a subset of frames from the corresponding key in the data (.npz) file.
    """
    with np.printoptions(precision=2, suppress=True):
        fig, ax = plt.subplots()
        ax.set_title(
            f"{key} frames from feature index {feature_idx}: {frame_start} - {frame_end}")
        ax.set_xlabel("frames")
        ax.set_ylabel("n_bootstraps")
        coef = data[key][:24, frame_start:frame_end, feature_idx]
        plot = ax.imshow(coef, interpolation='none',
                         extent=[0, 24, 24, 0], vmin=-15, vmax=15)
        plt.figtext(0, 0, str(coef[:24, frame_start:frame_end]), fontsize=6)
        fig.colorbar(plot, ax=ax)
        plt.show()
        plt.close()


def graph_2d_support_matrix(support_matrix: np.ndarray, filename: str) -> None:
    """
    Graph a 2d support matrix in a plot.
    """
    fig, ax = plt.subplots()
    ax.set_title(f"2d support matrix for run {filename}")
    ax.set_xlabel("features")
    ax.set_ylabel("regularization strength")
    plot = ax.imshow(support_matrix, interpolation='none')
    colors = [(val, plot.cmap(plot.norm(val))) for val in [True, False]]
    patches = [mpatches.Patch(color=color, label=val) for val, color in colors]
    y_ticks, x_ticks = support_matrix.shape
    ax.set_xticks(np.arange(0, x_ticks, 1))
    ax.set_yticks(np.arange(0, y_ticks, 1))

    plt.legend(handles=patches)
    plt.show()
    plt.close()


def main(parsed_args: argparse.Namespace):
    """Prints the first and last 10 results from the file and generates a graph of results"""

    file = parsed_args.input_file
    key = parsed_args.key

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
        with np.load(file, allow_pickle=True) as data:
            for key in data.keys():
                if key == "x_coefficients":
                    for i in range(5):
                        graph_2d_subset_x_linear_classification_coefficients(
                            key="x_coefficients", data=data, frame_start=0, frame_end=1000, feature_idx=i)
    elif file.endswith(".json"):
        with open(file) as data:
            attributes = json.load(data)
            if key == "supports_":
                # The numpy matrices are flattened, so they have to be reshaped at read time.
                shape = attributes[key][0]
                base64_array = base64.b64decode(
                    attributes[key][1])
                array = np.frombuffer(base64_array, dtype="bool").reshape(shape)
                graph_2d_support_matrix(support_matrix=array, filename=file)


if __name__ == "__main__":
    arg_parser = initialize_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    main(parsed_args)
