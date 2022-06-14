import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tpr(expected: np.array, observed: np.array) -> int:
    assert len(expected) == len(observed)
    true_positives = sum(e == 1 and o == 1 for e, o in zip(expected, observed))
    false_negatives = sum(e == 1 and o == 0 for e, o in zip(expected, observed))
    return true_positives / (true_positives + false_negatives)


def fpr(expected: np.array, observed: np.array) -> int:
    assert len(expected) == len(observed)
    false_positives = sum(e == 0 and o == 1 for e, o in zip(expected, observed))
    true_negatives = sum(e == 0 and o == 0 for e, o in zip(expected, observed))
    return false_positives / (false_positives + true_negatives)


def auc(fpr: np.array, tpr: np.array) -> np.array:
    pass


def plot_roc_curve(expected: np.array, observed: np.array, label: int, confidence: np.array):
    df = pd.DataFrame()
    df['confidence'] = confidence
    expected = [1 if e == label else 0 for e in expected]
    observed = [1 if o == label else 0 for o in expected]
    df['expected'] = expected
    df['observed'] = observed

    # Sort based on confidence values
    df.sort_values('confidence', ascending=False)

    # Accumulate values and calculate tpr and fpr rates per element
    df['tpr'] = 0
    df['fpr'] = 0
    for i in range(len(df)):
        df['tpr'] = df.iloc[i, ]
        df['']

    # Plot all points along threshold
    fig, ax = plt.subplots()


def main():
    plot_roc_curve(np.array([1, 1, 0, 0]), np.array(
        [1, 1, 0, 1]), 1, np.array([0, 0.2, 0.4, 0.5]))


if __name__ == '__main__':
    main()
