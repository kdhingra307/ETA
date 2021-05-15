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
from tensorflow.keras.callbacks import TensorBoard
from os.path import join as join_directory


optimizer = tf_keras.optimizers.Adam(
    learning_rate=config.training.learning_rate
)
model = Model()
model.compile(
    optimizer=optimizer,
    loss={
        "ind/mse": loss_function,
        "ind/generator": tf_keras.losses.BinaryCrossentropy(from_logits=True),
        "ind/discriminator": tf_keras.losses.BinaryCrossentropy(
            from_logits=True
        ),
    },
    metrics=metrics,
)

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

ckpt_manager = CheckpointManager(optimizer, model, ckpt_dir)
log_manager = TensorBoard(
    log_dir=log_dir, update_freq="batch", histogram_freq=1, embeddings_freq=5
)
ckpt_manager.ckpt_manager.restore_or_initialize()
optimizer.learning_rate = config.training.learning_rate

model.fit(
    Dataset(train_split),
    epochs=config.training.epochs,
    callbacks=[ckpt_manager, log_manager],
    validation_data=Dataset(validation_split),
    # verbose=2,
)
