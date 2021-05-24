import numpy as np


def mse(y_true, y_pred):
    return np.sqrt(np.nanmean(np.power(y_true - y_pred, 2)))
