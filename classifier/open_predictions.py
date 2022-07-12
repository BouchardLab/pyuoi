import argparse
from tkinter import N
from turtle import color
import numpy as np
from numpy.lib.npyio import NpzFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay, auc, PrecisionRecallDisplay, roc_curve
from sklearn.svm import LinearSVC
import pickle
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
    parser.add_argument('--output_graphs',
                        help="Output matplotlib graphs from run",
                        default=False, type=bool)
    return parser


def read_numpy_binary_array(attributes: dict, key: str, dtype: np.dtype) -> np.array:
    """
    Read in numpy array from binary file format. The numpy matrices are flattened, so they have to be reshaped at read time.
    """
    shape = attributes[key][0]
    base64_array = base64.b64decode(
        attributes[key][1])
    array = np.frombuffer(base64_array, dtype=dtype).reshape(shape)
    return array


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


def main(parsed_args: argparse.Namespace, debug: bool = False):
    """
    Prints the first and last 10 results from the file and generates a graph of results. The following commands are meant to be used as examples.

    >>> python classifier/open_predictions.py --input_file /Users/josephgmaa/pyuoi/pyuoi/data/features/run_parameters/20220712-142558.run_parameters.json --key="supports_"

    >>> python classifier/open_predictions.py --input_file /Users/josephgmaa/pyuoi/pyuoi/data/features/run_parameters/20220712-142558.run_parameters.json --key="accuracy"
    """

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
                array = read_numpy_binary_array(
                    attributes=attributes, key=key, dtype=bool)
                graph_2d_support_matrix(support_matrix=array, filename=file)
            elif key == "metrics":
                expected = read_numpy_binary_array(
                    attributes=attributes, key="expected_output", dtype=np.uint64)

                predicted = read_numpy_binary_array(
                    attributes=attributes, key="predicted_output", dtype=np.uint64)

                ConfusionMatrixDisplay.from_predictions(expected, predicted)
                plt.show()
            elif key == "selection_thresholds_":
                array = read_numpy_binary_array(
                    attributes=attributes, key=key, dtype=np.uint8)
            elif key == "accuracy":
                print("Reading binary arrays.")
                expected = read_numpy_binary_array(
                    attributes=attributes, key="expected_output", dtype=np.int64)

                predicted = read_numpy_binary_array(
                    attributes=attributes, key="predicted_output", dtype=np.int64)

                prediction_probabilities = read_numpy_binary_array(
                    attributes=attributes, key="prediction_probabilities", dtype=np.float64)

                with open(attributes['label_encoder'], 'rb') as label_encoder_pickle_file:
                    label_encoder = pickle.load(label_encoder_pickle_file)

                classes = np.unique(expected)
                n_classes = len(np.unique(expected))

                x = label_binarize(expected, classes=classes)
                y = label_binarize(predicted, classes=classes)

                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.33, random_state=0)

                probabilities_x_train, probabilities_x_test, probabilities_y_train, probabilities_y_test = train_test_split(
                    x, prediction_probabilities, test_size=0.33, random_state=0)

                # Compute ROC curve and ROC area for each class.
                fpr, tpr, roc_auc = {}, {}, {}
                print(f'There are {n_classes} classes')
                for i in range(n_classes):
                    print(f'predicted instances: {y_test[:, i]}')
                    print(f'probabilities of y {y_test[:, i]}')
                    plt.figure()
                    plt.title(
                        'Predicted instances of simulated data from a single class')
                    plt.bar(range(len(y_test)), y_test[:, i])
                    plt.show()
                    plt.figure()
                    plt.title(
                        'Predicted probabilities for simulated data from a single class')
                    plt.bar(range(len(y_test)), probabilities_y_test[:, i])
                    plt.show()
                    if debug:
                        with np.printoptions(threshold=np.inf):
                            print(fpr, tpr)
                    fpr[i], tpr[i], thresholds = roc_curve(
                        y_test[:, i], probabilities_y_test[:, i], drop_intermediate=False)
                    print(f'Thresholds: {thresholds}')
                    plt.figure()
                    plt.title(
                        'Automatically generated thresholds by sklearn')
                    plt.bar(range(len(y_test) + 1), thresholds)
                    plt.show()
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    print(f'fpr: {fpr} \ntpr: {tpr}')
                    plt.figure()
                    plt.title(
                        'FPR and TPR calculations for simulated data from a single class')
                    plt.bar(range(len(y_test) + 1),
                            fpr[i], color='r', alpha=0.5, label="fpr")
                    plt.bar(range(len(y_test) + 1),
                            tpr[i], color='g', alpha=0.5, label="tpr")
                    plt.legend()
                    plt.show()
                    break
                if not parsed_args.output_graphs:
                    ConfusionMatrixDisplay.from_predictions(expected, predicted)
                    plt.show()

                # Plot of a ROC curve for a specific class
                for i in range(n_classes):
                    # Display only if the class is predicted at least once
                    if sum(y_test[:, i] >= 1):
                        plt.figure()
                        plt.plot(fpr[i], tpr[i],
                                 label='ROC curve from pyuoi (area = %0.5f)' % roc_auc[i])
                        plt.plot([0, 1], [0, 1], 'k--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        # TODO(Joseph): Add the features that generate the ROC curves
                        plt.title(
                            f'Receiver operating characteristic {label_encoder.inverse_transform(np.array([classes[i]]))}')
                        plt.legend(loc="lower right")
                        if parsed_args.output_graphs:
                            print(
                                f"Save ROC curves to: {os.path.join('/Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/roc_curves', os.path.basename(parsed_args.input_file))}_{i}_ROC_curve.png")
                            plt.savefig(
                                f"{os.path.join('/Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/roc_curves', os.path.basename(parsed_args.input_file))}_{i}_ROC_curve.png")
                        else:
                            plt.show()
                        break


if __name__ == "__main__":
    arg_parser = initialize_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    main(parsed_args)
