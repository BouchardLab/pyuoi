import numpy as np
import sys
import logging
import os
import time
import json
import base64


def softmax(y, axis=-1):
    """Calculates the softmax distribution.

    Parameters
    ----------
    y : ndarray
        Log-probabilities.
    """

    yp = y - y.max(axis=axis, keepdims=True)
    epy = np.exp(yp)
    return epy / np.sum(epy, axis=axis, keepdims=True)


def sigmoid(x):
    """Calculates the bernoulli distribution.

    Parameters
    ----------
    x : ndarray
        Log-probabilities.
    """
    return np.exp(-np.logaddexp(0, -x))


def log_likelihood_glm(model, y_true, y_pred):
    """Calculates the log-likelihood of a generalized linear model given the
    true response variables and the "predicted" response variables. The
    "predicted" response variable varies by the specific generalized linear
    model under consideration.

    Parameters
    ----------
    model : string
        The generalized linear model to calculate the log-likelihood for.
    y_true : nd-array, shape (n_samples,)
        Array of true response values.
    y_pred : nd-array, shape (n_samples,)
        Array of predicted response values (conditional mean).

    Returns
    -------
    ll : float
        The log-likelihood.
    """
    if model == 'normal':
        # this log-likelihood is calculated under the assumption that the
        # variance is the value that maximizes the log-likelihood
        rss = (y_true - y_pred)**2
        n_samples = y_true.size
        ll = -n_samples / 2 * (1 + np.log(np.mean(rss)))
    elif model == 'poisson':
        if not np.any(y_pred):
            if np.any(y_true):
                ll = -np.inf
            else:
                ll = 0.
        else:
            ll = np.mean(y_true * np.log(y_pred) - y_pred)
    else:
        raise ValueError('Model is not available.')
    return ll


def BIC(ll, n_features, n_samples):
    """Calculates the Bayesian Information Criterion.

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.
    n_features : int
        The number of features used in the model.
    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    BIC : float
        Bayesian Information Criterion
    """
    BIC = n_features * np.log(n_samples) - 2 * ll
    return BIC


def AIC(ll, n_features):
    """Calculates the Akaike Information Criterion.

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.
    n_features : int
        The number of features used in the model.
    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    AIC : float
        Akaike Information Criterion
    """

    AIC = 2 * n_features - 2 * ll
    return AIC


def AICc(ll, n_features, n_samples):
    """Calculate the corrected Akaike Information Criterion. This criterion is
    useful in cases when the number of samples is small.

    If the number of features is equal to the number of samples plus one, then
    the AIC is returned (the AICc is undefined in this case).

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.
    n_features : int
        The number of features used in the model.
    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    AIC : float
        Akaike Information Criterion
    """
    AICc = AIC(ll, n_features)
    if n_samples > (n_features + 1):
        AICc += 2 * (n_features**2 + n_features) / (n_samples - n_features - 1)
    return AICc


def check_logger(logger, name='uoi', comm=None):
    ret = logger
    if ret is None:
        if comm is not None and comm.Get_size() > 1:
            r, s = comm.Get_rank(), comm.Get_size()
            name += " " + str(r).rjust(int(np.log10(s)) + 1)

        ret = logging.getLogger(name=name)
        handler = logging.StreamHandler(sys.stdout)

        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        handler.setFormatter(logging.Formatter(fmt))
        ret.addHandler(handler)
    return ret


def is_json_serializable(object: any) -> bool:
    try:
        json.dumps(object)
        return True
    except (TypeError, OverflowError):
        return False


def dump_json(model: "AbstractUoILinearModel", filename: str, results: dict) -> None:
    """
    Writes out all model values to JSON.
    """
    dirname, basename = os.path.dirname(filename), os.path.basename(
        filename)
    with open(generate_timestamp_filename(dirname=dirname, basename=basename, file_format=".json"), "w") as file:
        json_dump = {}
        for dump in (model.__dict__, results):
            for key, val in dump.items():
                if is_json_serializable(val):
                    json_dump[key] = val
                elif isinstance(val, np.ndarray):
                    # Encode arrays as base64 strings.
                    json_dump[key] = (val.shape, str(
                        base64.b64encode(val), 'utf-8'))
        print(
            f"JSON attributes written to {generate_timestamp_filename(dirname=dirname, basename=basename, file_format='.json')}.")
        json.dump(json_dump, file, sort_keys=True, indent=4)


def generate_timestamp_filename(dirname: str, basename: str, file_format: str) -> str:
    """
    Generate a timestamped filename for use in saving files.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(dirname, f"{timestr}.{basename}{file_format}")


def write_timestamped_numpy_binary(filename: str, **data: np.array) -> None:
    """
    Writes a numpy binary file with a timestamped prefix to a 'saved_runs' directory in the same directory.
    """

    basename, dirname = os.path.basename(filename), os.path.dirname(filename)

    saved_runs_directory = os.path.join(dirname, 'saved_runs')

    if not os.path.exists(saved_runs_directory):
        os.makedirs(saved_runs_directory)

    # Only process one keyword argument array.
    if len(data) == 1:
        saved_filename = generate_timestamp_filename(
            dirname=saved_runs_directory, basename=basename, file_format='.npy')

        np.save(saved_filename, data)

    # Process multiple keyworse argument arrays.
    else:
        saved_filename = saved_filename = generate_timestamp_filename(
            dirname=saved_runs_directory, basename=basename, file_format='.npz')

        np.savez(saved_filename, **data)

    print('File saved to: ', saved_filename)
