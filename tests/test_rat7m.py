from pyuoi.linear_model import UoI_L1Logistic
from pyuoi.datasets import make_classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import xarray as xr

from sklearn.model_selection import train_test_split

def main():
    start = time.time()
    filename = "/Users/josephgmaa/pyuoi/pyuoi/data/nolj_Recording_day7_overnight_636674151185633714_1_nolj.c3d.243.features.netcdf"

    df = xr.load_dataset(filename).to_dataframe()

    rows, columns = df.shape

    # Use the an1 columns with distance for the first pass.
    indices = df.columns
    # print(an1_indices)
    # print([column for i, column in enumerate(
    #     df.columns) if column.startswith('distance_an1')])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # X = df.loc[:, 'egocentric_an1_relative_velocity_an1_BodyCenter_x'].to_numpy()
    X = df.iloc[:, indices].to_numpy()
    print(X)
    # Binarize ProneStill
    map_prone_vs_others = {'Walk': 0, 'WetDogShake': 0, 'FaceGroom': 0, 'RScratch': 0, 'BadTracking': 0, 'RGroom': 0,
                            'ProneStill': 1, 'AdjustPosture': 0}
    # y = df['behavior_name'].map(map_prone_vs_others)
    y = df['behavior_name'].to_numpy()
    # print([column for column in df.columns.tolist() if 'velocity' in column])

    fig, ax = plt.subplots()
    ax.plot(y)
    plt.show()

    # for feature in df.columns:
    # map_behavior_to_values = {'Walk': 100, 'WetDogShake': 200, 'FaceGroom': 300, 'RScratch': 400, 'BadTracking': 500, 'RGroom': 600,
                            # 'ProneStill': 700, 'AdjustPosture': 800}
    # df['behavior_values'] = df['behavior_name'].map(map_behavior_to_values)
    # ax.plot(df['egocentric_an1_relative_velocity_an1_BodyCenter_x'])
    # ax.scatter(df.index, df.behavior_values)
    # ax.text(x=0,y=-100,s=[f"{key}: {value}" for key, value in map_behavior_to_values.items()], fontsize="small")
    # plt.show()

    # ax.plot(df['egocentric_an1_relative_velocity_an1_HeadCenter_y'])
    # ax.plot(y)
    # plt.show()

    # # Run the classifier.
    # print('Running the classifier.')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # l1log = UoI_L1Logistic().fit(X_train, y_train)
    # y_hat = l1log.predict(X_test)
    # print(y_hat)

    # # Save the results to a numpy binary.
    # with open('test.npy', 'wb') as file:
    #     np.save(file, X)
    #     np.save(file, y)
    #     np.save(file, y_hat)
    #     np.save(file, y_test)

    # print(np.load('test.npy').shape)

    # ax.clear()
    # ax.plot(y_hat)
    # plt.show()

    end = time.time()
    print('Time elapsed: ', end - start)


if __name__ == "__main__":
    main()
