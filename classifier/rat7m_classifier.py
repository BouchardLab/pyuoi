from pyuoi import UoI_L1Logistic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyuoi.utils import write_timestamped_numpy_binary, dump_json
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
import xarray as xr
import numpy as np
import pandas as pd
import argparse
import sys


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
    return parser


def main(parsed_args: argparse.Namespace):
    """
    Run argument parser with commands like:

    >>> python classifier/rat7m_classifier.py --input_files /Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/PCs-mavg-velocity_relative.netcdf PCA_mavg_velocity_relative_0

    >>> python classifier/rat7m_classifier.py --input_files /Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/PCs-mavg-velocity_relative.netcdf --column_names PCA_mavg_velocity_relative_0 PCA_mavg_velocity_relative_1 PCA_mavg_velocity_relative_2 PCA_mavg_velocity_relative_3 PCA_mavg_velocity_relative_4
    """
    df = pd.DataFrame()
    for filename in parsed_args.input_files:
        df = pd.concat([df, xr.load_dataset(
            filename, engine='h5netcdf').to_dataframe()])
        # print(df['behavior_name'].value_counts())
        # print('shape', df.shape)
        # print(sorted(list(df.columns)))

    row_indices = np.arange(start=0, stop=df.shape[0]).tolist()
    df = df.iloc[row_indices, :]
    column_indices = [
        i for i in df.columns if any(i in word for word in parsed_args.column_names)]
    y = df['behavior_name'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        df.loc[:, column_indices].to_numpy(), y, random_state=parsed_args.training_seed)

    assert x_train.shape[
        1] > 0, f"X train dataset should have at least 1 input feature. Try checking that the column names match the input dataset: {list(df.columns)}"

    l1log = UoI_L1Logistic(random_state=parsed_args.model_seed, multi_class='multinomial').fit(
        x_train, y_train, verbose=True)

    y_hat = l1log.predict(x_test)
    accuracy = accuracy_score(y_test, y_hat)
    print('y: ', y)
    print('y_hat: ', y_hat)
    y_freq, y_hat_freq = np.bincount(y), np.bincount(y_hat)
    print('y_freq: ', y_freq)
    print('y_hat_freq: ', y_hat_freq)

    kl_divergence_y_hat_given_y = rel_entr(y_hat_freq, y_freq)
    kl_divergence_y_given_y_hat = rel_entr(y_freq, y_hat_freq)
    jensenshannon_predictions = jensenshannon(y_freq, y_hat_freq)

    if jensenshannon(y_freq, y_hat_freq) != jensenshannon(y_hat_freq, y_freq):
        print("Values are not symmetrical for jensenshannon predictions")

    print(f"Accuracy: {accuracy}")
    print(f"Resulting values: {y_hat}")

    if parsed_args.dump:
        dump_json(model=l1log, filename="/Users/josephgmaa/pyuoi/pyuoi/data/features/run_parameters/run_parameters",
                  results={"accuracy": accuracy,
                           "predicted_output": y_hat,
                           "kl_divergence_y_hat_given_y": kl_divergence_y_hat_given_y,
                           "kl_divergence_y_given_y_hat": kl_divergence_y_given_y_hat,
                           "jensenshannon_predictions":
                           jensenshannon_predictions,
                           "input_files": parsed_args.input_files,
                           "column_names": parsed_args.column_names, "train_test_split_seed": parsed_args.training_seed, "l1log_seed": parsed_args.model_seed})

    write_timestamped_numpy_binary(filename=filename, data=y_hat)


if __name__ == "__main__":
    arg_parser = initialize_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    main(parsed_args=parsed_args)
