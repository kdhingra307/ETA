from ETA import Model, Dataset, loss_function, metrics, config, CheckpointManager
from datetime import datetime
import tensorflow.keras as tf_keras
from tensorflow.keras.callbacks import TensorBoard
from os.path import join as join_directory


optimizer = tf_keras.optimizers.Adam(learning_rate=config.training.learning_rate)
model = Model()
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=metrics)

test_split = config.data.split_prefix.format("test")


ckpt_dir = join_directory(
    config.model.working_dir,
    config.training.ckpt_dir.format(
        config.model.training_label))

ckpt_manager = CheckpointManager(optimizer, model, ckpt_dir)
ckpt_manager.ckpt_manager.restore_or_initialize()

print(model.evaluate(Dataset(test_split)))
