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
from tensorflow import summary as tf_summary
from tensorflow.config import experimental as tf_gpu


gpus = tf_gpu.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf_gpu.set_memory_growth(gpu, True)
        logical_gpus = tf_gpu.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


optimizer = tf_keras.optimizers.Adam(
    learning_rate=config.training.learning_rate
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
    if epoch >= 10 and epoch <= 50 and epoch % 10 == 6:
        lr *= 0.5

    print(tf_summary.scalar("LearningRate", data=lr))
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
    initial_epoch=0,
    callbacks=[ckpt_manager, log_manager, lr_manager],
    validation_data=Dataset(validation_split),
)
