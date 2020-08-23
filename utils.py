from tensorflow.keras import backend as K
import numpy as np


def psnr(y_true, y_pred):
    return -10 * K.log(K.mean(K.flatten((y_true - y_pred)) ** 2)) / np.log(10)
