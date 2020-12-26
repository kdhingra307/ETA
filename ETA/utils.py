import yaml
from collections import namedtuple
from munch import munchify
import tensorflow.keras as tf_keras
import tensorflow.train as tf_train
from tensorflow import math as tf_maths
from tensorflow import where as tf_where
from datetime import datetime
from numpy import inf


def get_config(config_filename):
    data = yaml.safe_load(open(config_filename).read())
    munched = munchify(data)
    munched.model.temporal_label = datetime.now().strftime("%Y%m%d-%H%M%S")
    return munched


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

class CheckpointManager(tf_keras.callbacks.Callback):
    
    def __init__(self, optimizer, model, ckpt_dir, label="val_loss"):

        self.loss = inf
        self.label = label
        checkpoint = tf_train.Checkpoint(optimizer=optimizer, model=model)
        self.ckpt_manager = tf_train.CheckpointManager(checkpoint=checkpoint,
                                                       directory=ckpt_dir, max_to_keep=5)
    

    def on_epoch_end(self, epoch, logs):
        if logs[self.label] < self.loss:
            self.ckpt_manager.save()