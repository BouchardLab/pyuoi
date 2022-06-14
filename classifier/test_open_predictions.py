import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def main():
    expected = np.array([1, 0, 0, 1, 0, 1])
    predicted = np.array([0, 0, 1, 0, 1, 1])

    fpr, tpr, _ = roc_curve(expected, predicted)
    print(fpr, tpr)

    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve from pyuoi (area = %0.2f)' % roc_auc)
    plt.show()


if __name__ == "__main__":
    main()
