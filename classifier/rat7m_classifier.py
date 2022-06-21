from sklearn.preprocessing import LabelEncoder
from pyuoi import UoI_L1Logistic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyuoi import datasets
from pyuoi.utils import write_timestamped_numpy_binary, dump_json, generate_timestamp_filename
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from pyuoi.datasets import make_classification
from itertools import combinations
import matplotlib.pyplot as plt
import os
import pickle
import xarray as xr
import numpy as np
import pandas as pd
import argparse
import sys


def absolute_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def initialize_arg_parser():
    parser = argparse.ArgumentParser(
        description='Read in numpy files and visualize frequency differences')
    parser.add_argument('--input_files',
                        help='Path to the input files.',
                        default=[
                            "/Users/josephgmaa/pyuoi/pyuoi/data/features/nolj_Recording_day7_overnight_636674151185633714_5_nolj.c3d.1851.features.netcdf"],
                        nargs='+')
    parser.add_argument('--column_names',
                        help='The column names to be used for parsing',
                        default="PCA_mavg_velocity_relative_0",
                        nargs='+')
    parser.add_argument("--dump",
                        help="A boolean flag whether to dump the results to a JSON file.",
                        default=True)
    parser.add_argument("--training_seed",
                        help="Seed for setting train_test_split.",
                        default=10, type=int)
    parser.add_argument("--model_seed",
                        help="Seed for setting internal model random state.",
                        default=10, type=int)
    parser.add_argument("--use_small_dataset",
                        help="Whether to use a small dataset for debugging",
                        default=False, type=bool)
    parser.add_argument("--grid_search",
                        help="Grid search through all the netcdfs in the directory. Takes the first 5 columns as values to automate the search.", default="/Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/", type=str)
    return parser


def main(parsed_args: argparse.Namespace):
    """
    Run argument parser with commands:

    >>> python classifier/rat7m_classifier.py --input_files /Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/PCs-mavg-velocity_relative.netcdf --column_names PCA_mavg_velocity_relative_0

    >>> python classifier/rat7m_classifier.py --input_files /Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/PCs-mavg-velocity_relative.netcdf --column_names PCA_mavg_velocity_relative_0 PCA_mavg_velocity_relative_1 PCA_mavg_velocity_relative_2 PCA_mavg_velocity_relative_3 PCA_mavg_velocity_relative_4

    To run with a small dataset for debugging, pass in the flag --use_small_dataset:

    >>> python classifier/rat7m_classifier.py --use_small_dataset True

    >>> python classifier/rat7m_classifier.py --grid_search /Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/  
    """
    if parsed_args.use_small_dataset:
        n_features = 20
        n_inf = 10
        x, y, _, _ = make_classification(n_samples=1000,
                                         random_state=10,
                                         n_classes=5,
                                         n_informative=n_inf,
                                         n_features=n_features,
                                         shared_support=True,
                                         w_scale=4.)

        le = LabelEncoder()
        le.fit(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=parsed_args.training_seed)
    elif parsed_args.grid_search:
        # Load all netcdfs into dataframes from the parent directory.
        pc_netcdfs = [file for file in absolute_file_paths(
            parsed_args.grid_search) if file.endswith(".netcdf")]
        for files in combinations(pc_netcdfs, 3):
            df = pd.DataFrame()
            for i, filename in enumerate(files):
                dataset = xr.load_dataset(
                    filename, engine="h5netcdf").to_dataframe()
                if i == 0:
                    df['behavior_name'] = dataset['behavior_name']
                df = df.reset_index(drop=True)
                df = pd.concat([df, dataset.iloc[:, 0:5]], axis=1)
            y = df['behavior_name']
            df = df.iloc[::50, :]
            # df.plot()
            # plt.show()
            # Remove this break later to test all
            break

        y = df['behavior_name'].to_numpy()
        le = LabelEncoder()
        le.fit(y)

        x_train, x_test, y_train, y_test = train_test_split(
            df.loc[:, df.columns != 'behavior_name'].to_numpy(), le.transform(y), random_state=parsed_args.training_seed)

        l1log = UoI_L1Logistic(random_state=parsed_args.model_seed, multi_class='multinomial').fit(
            x_train, y_train, verbose=True)

        y_hat = l1log.predict(x_test)

        accuracy = accuracy_score(y_test, y_hat)
        print('y_test: ', y_test)
        print('y_hat: ', y_hat)
        y_test_freq = np.bincount(y_test)
        y_hat_freq = np.bincount(y_hat)
        print('y_test_freq: ', y_test_freq)
        print('y_hat_freq: ', y_hat_freq)
        return
    else:
        df = pd.DataFrame()
        for filename in parsed_args.input_files:
            df = pd.concat([df, xr.load_dataset(
                filename, engine='h5netcdf').to_dataframe()])

        row_indices = np.arange(start=0, stop=df.shape[0]).tolist()
        df = df.iloc[row_indices, :]
        column_indices = [
            i for i in df.columns if any(i in word for word in parsed_args.column_names)]

        y = df['behavior_name'].to_numpy()
        le = LabelEncoder()
        le.fit(y)

        x_train, x_test, y_train, y_test = train_test_split(
            df.loc[:, column_indices].to_numpy(), le.transform(y), random_state=parsed_args.training_seed)

    assert x_train.shape[
        1] > 0, f"X train dataset should have at least 1 input feature. Try checking that the column names match the input dataset: {list(df.columns)}"

    l1log = UoI_L1Logistic(random_state=parsed_args.model_seed, multi_class='multinomial').fit(
        x_train, y_train, verbose=True)

    y_hat = l1log.predict(x_test)

    accuracy = accuracy_score(y_test, y_hat)
    print('y_test: ', y_test)
    print('y_hat: ', y_hat)
    y_test_freq = np.bincount(y_test)
    y_hat_freq = np.bincount(y_hat)
    print('y_test_freq: ', y_test_freq)
    print('y_hat_freq: ', y_hat_freq)

    kl_divergence_y_hat_given_y = rel_entr(y_hat_freq, y_test_freq)
    kl_divergence_y_given_y_hat = rel_entr(y_test_freq, y_hat_freq)
    jensenshannon_predictions = jensenshannon(y_test_freq, y_hat_freq)

    print(f"Accuracy: {accuracy}")
    print(f"Resulting values: {y_hat}")

    filename = "/Users/josephgmaa/pyuoi/pyuoi/data/features/run_parameters/run_parameters"

    if parsed_args.dump:
        # Dump the associated label encoder.
        le_filename = generate_timestamp_filename(
            dirname=os.path.dirname(filename), basename="label_encoder", file_format=".pkl")
        with open(le_filename, 'wb+') as f:
            pickle.dump(le, f)
        print(
            f'Label encoder written to {le_filename}.')

        dump_json(model=l1log, filename=filename,
                  results={"accuracy": accuracy,
                           "coefficients": l1log.coef_,
                           "expected_output": y_test,
                           "predicted_output": y_hat,
                           "label_encoder": le,
                           "kl_divergence_y_hat_given_y": kl_divergence_y_hat_given_y,
                           "kl_divergence_y_given_y_hat": kl_divergence_y_given_y_hat,
                           "jensenshannon_predictions":
                           jensenshannon_predictions,
                           "prediction_probabilities":
                           l1log.predict_proba(x_test),
                           "input_files": parsed_args.input_files,
                           "column_names": parsed_args.column_names,
                           "label_encoder": le_filename,
                           "train_test_split_seed": parsed_args.training_seed, "l1log_seed": parsed_args.model_seed})

    write_timestamped_numpy_binary(filename=filename, data=y_hat)


if __name__ == "__main__":
    arg_parser = initialize_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    main(parsed_args=parsed_args)
