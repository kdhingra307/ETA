import pickle
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


class CheckpointManager(tf_keras.callbacks.Callback):
    def __init__(self, optimizer, model, ckpt_dir, label="val_loss"):

        self.loss = inf
        self.label = label
        checkpoint = tf_train.Checkpoint(optimizer=optimizer, model=model)
        self.ckpt_manager = tf_train.CheckpointManager(
            checkpoint=checkpoint, directory=ckpt_dir, max_to_keep=5
        )

    def on_epoch_end(self, epoch, logs):
        if logs[self.label] < self.loss:
            self.ckpt_manager.save()
