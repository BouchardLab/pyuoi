from pyuoi import UoI_L1Logistic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xarray as xr
import numpy as np
import os
import time

def main():
    filename = "/Users/josephgmaa/pyuoi/pyuoi/data/features/nolj_Recording_day7_overnight_636674151185633714_5_nolj.c3d.1851.features.netcdf"

    df = xr.load_dataset(filename, engine='h5netcdf').to_dataframe()

    # Use only the egocentric relative velocities for the training
    row_indices = np.arange(start=0, stop=df.shape[0]).tolist()
    df = df.iloc[row_indices, :]
    column_indices = [
        i for i in df.columns if "egocentric_an1_relative_velocity" in i]
    y = df['behavior_name'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        df.loc[:, column_indices].to_numpy(), y)

    l1log = UoI_L1Logistic(random_state=10, multi_class='multinomial').fit(
        x_train, y_train, verbose=True)
    y_hat = l1log.predict(x_test)
    print('Accuracy: ', accuracy_score(y_test, y_hat))
    print('Resulting values: ', y_hat)

    basename, dirname = os.path.basename(filename), os.path.dirname(filename)

    saved_runs_directory = os.path.join(dirname, 'saved_runs')

    if not os.path.exists(saved_runs_directory):
        os.makedirs(saved_runs_directory)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    saved_filename = os.path.join(
        saved_runs_directory, timestr + '.' + basename + '.npy')

    np.save(saved_filename, y_hat)

    print('File saved to: ', saved_filename)


if __name__ == "__main__":
    main()
