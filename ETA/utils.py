import yaml
from collections import namedtuple
from munch import munchify
import tensorflow.keras as tf_keras
from tensorflow import math as tf_maths
from tensorflow import where as tf_where

def get_config(config_filename):
    data = yaml.safe_load(open(config_filename).read())
    return munchify(data)

def mse(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]
    output = ((y_true - y_pred)**2) * mask
    return tf_maths.reduce_sum(output)/tf_maths.reduce_sum(mask)

def mae(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]
    output = (tf_maths.abs(y_true - y_pred)) * mask
    return tf_maths.reduce_sum(output)/tf_maths.reduce_sum(mask)

def rmse(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]
    output = ((y_true - y_pred)**2) * mask
    return tf_maths.sqrt(tf_maths.reduce_sum(output)/tf_maths.reduce_sum(mask))

def mape(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    y_true = y_true[:, :, :, 0]
    output = tf_maths.abs(y_true - y_pred) / y_true
    output = tf_where(tf_maths.is_nan(output), mask, output)
    output = tf_where(tf_maths.is_inf(output), mask, output)

    return tf_maths.reduce_sum(output)/tf_maths.reduce_sum(mask)