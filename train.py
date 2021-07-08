from ETA import (
    Model,
    Dataset,
    loss_function,
    metrics,
    config,
    CheckpointManager,
)
from datetime import datetime
import tensorflow.keras as tf_keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from os.path import join as join_directory
import tensorflow as tf


optimizer = tf_keras.optimizers.Adagrad(
    learning_rate=config.training.learning_rate, epsilon=1e-3
)
model = Model()
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

train_split = config.data.split_prefix.format("train")
validation_split = config.data.split_prefix.format("val")
test_split = config.data.split_prefix.format("test")

log_dir = join_directory(
    config.model.working_dir,
    config.training.log_dir.format(config.model.training_label),
)

ckpt_dir = join_directory(
    config.model.working_dir,
    config.training.ckpt_dir.format(config.model.training_label),
)


def scheduler(epoch, lr):
    if epoch >= 15 and epoch <= 50 and epoch % 8 == 0:
        lr *= 0.8

    print(tf.summary.scalar("LearningRate", data=lr))
    return lr


ckpt_manager = CheckpointManager(optimizer, model, ckpt_dir)
log_manager = TensorBoard(
    log_dir=log_dir, update_freq="batch", histogram_freq=1, embeddings_freq=5
)
lr_manager = LearningRateScheduler(scheduler)
ckpt_manager.ckpt_manager.restore_or_initialize()

model.fit(
    Dataset(train_split),
    epochs=config.training.epochs,
    callbacks=[ckpt_manager, log_manager, lr_manager],
    validation_data=Dataset(validation_split),
)
