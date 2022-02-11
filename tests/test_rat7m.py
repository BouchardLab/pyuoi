from pyuoi.linear_model import UoI_L1Logistic
from pyuoi.datasets import make_classification
from matplotlib.widgets import Button


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import xarray as xr

from sklearn.model_selection import train_test_split

# filename = "/Users/josephgmaa/pyuoi/pyuoi/data/nolj_Recording_day7_overnight_636674151185633714_1_nolj.c3d.243.features.netcdf"

filename = "/Users/josephgmaa/pyuoi/pyuoi/data/features/nolj_Recording_day7_overnight_636674151185633714_35_nolj.c3d.916.features.netcdf"

df = xr.load_dataset(filename, engine='h5netcdf').to_dataframe()

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)


class Index(object):
    global df  # used so you can access local list, funcs, here
    global ax

    def __init__(self):
        self.ind = 0
        self.line = None

    def next(self, event):
        if self.line:
            self.line.pop(0).remove()
            ax.clear()
        ax.plot(df['behavior_name'].to_numpy(), alpha=0.5)
        self.ind += 1
        y = df.iloc[:, self.ind].to_numpy()
        name = df.columns[self.ind]
        self.line = ax.plot(y, alpha=0.5)  # set y value data
        ax.title.set_text(name)  # set title of graph
        plt.draw()

    def prev(self, event):
        if self.line:
            self.line.pop(0).remove()
            ax.clear()
        ax.plot(df['behavior_name'].to_numpy(), alpha=0.5)
        self.ind += 1
        y = df.iloc[:, self.ind].to_numpy()
        name = df.columns[self.ind]
        self.line = ax.plot(y, alpha=0.5)  # set y value data
        ax.title.set_text(name)  # set title of graph
        plt.draw()


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    start = time.time()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    ax.plot(df['behavior_name'].to_numpy())
    plt.show()

    # for feature in df.columns:
    #     map_behavior_to_values = {'Walk': 100, 'WetDogShake': 200, 'FaceGroom': 300, 'RScratch': 400, 'BadTracking': 500, 'RGroom': 600,
    #                               'ProneStill': 700, 'AdjustPosture': 800}
    #     df['behavior_values'] = df['behavior_name'].map(map_behavior_to_values)
    #     feature = ax.plot(df[feature])
    #     ax.scatter(df.index, df.behavior_values)
    #     ax.text(x=0, y=-100, s=[f"{key}: {value}" for key,
    #             value in map_behavior_to_values.items()], fontsize="small")
    #     plt.show()
    #     del feature

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
