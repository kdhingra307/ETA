from tensorflow import math as tf_maths
from tensorflow import where as tf_where
from ETA import config
import numpy as np
import tensorflow as tf

mean, std = config.data.mean, config.data.std
mean = mean[0]
std = std[0]


def mse(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    y_true = (y_true*std + mean)
    y_pred = (y_pred*std + mean)

    output = ((y_true - y_pred)**2) * mask
    return tf_maths.reduce_sum(output)/tf_maths.reduce_sum(mask)


def mae(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    y_true = (y_true*std + mean)
    y_pred = (y_pred*std + mean)

    output = (tf_maths.abs(y_true - y_pred)) * mask
    return tf_maths.reduce_sum(output)/tf_maths.reduce_sum(mask)


def rmse(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    y_true = (y_true*std + mean)
    y_pred = (y_pred*std + mean)

    output = ((y_true - y_pred)**2) * mask
    return tf_maths.sqrt(tf_maths.reduce_sum(output)/tf_maths.reduce_sum(mask))


def mape(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    y_true = (y_true*std + mean)
    y_pred = (y_pred*std + mean)

    print(tf.reduce_min(y_true))

    output = tf_maths.abs(y_true - y_pred) / y_true
    output = tf_where(tf_maths.is_nan(output), mask, output)
    output = tf_where(tf_maths.is_inf(output), mask, output)

    return tf_maths.reduce_sum(output)/tf_maths.reduce_sum(mask)