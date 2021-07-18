from tensorflow import math as tf_maths
from tensorflow import where as tf_where
from ETA import config
import numpy as np
import tensorflow as tf
import sys

mean, std = config.data.mean, config.data.std
mean = mean[0]
std = std[0]


def loss(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    y_true = y_true * std + mean
    y_pred = y_pred * std + mean

    # y_pred = tf_maths.log(tf_maths.maximum(y_pred, 1e-7) + 1.0)
    # y_true = tf_maths.log(tf_maths.maximum(y_true, 1e-7) + 1.0)

    output = (
        tf.maximum(0.55 * (y_true - y_pred), 0.45 * (y_pred - y_true)) * mask
    )
    # output = (y_true - y_pred) ** 2 * mask

    # output = tf.where(y_true > y_pred, 1.2 * output, 1 * output)

    return tf_maths.reduce_sum(output) / tf_maths.reduce_sum(mask)


def mse(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    output = ((y_true - y_pred) ** 2) * mask
    return tf_maths.reduce_sum(output) / tf_maths.reduce_sum(mask)


def mae(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    output = (tf_maths.abs(y_true - y_pred)) * mask
    return tf_maths.reduce_sum(output) / tf_maths.reduce_sum(mask)


def direction(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    y_true = y_true * std + mean
    y_pred = y_pred * std + mean

    output = tf.cast(y_true <= y_pred, tf.float32) * mask
    return tf_maths.reduce_sum(output) / tf_maths.reduce_sum(mask)


def rmse(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    output = ((y_true - y_pred) ** 2) * mask
    return tf_maths.sqrt(
        tf_maths.reduce_sum(output) / tf_maths.reduce_sum(mask)
    )


def mape(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]

    output = tf_maths.abs(y_true - y_pred) / tf_maths.abs(y_true)
    output = tf_where(tf_maths.is_nan(output), mask, output)
    output = tf_where(tf_maths.is_inf(output), mask, output)

    return tf_maths.reduce_sum(output) / tf_maths.reduce_sum(mask)
