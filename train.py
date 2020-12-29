from ETA import Model, Dataset, loss_function, metrics, config, CheckpointManager
from datetime import datetime
import tensorflow.keras as tf_keras
from tensorflow.keras.callbacks import TensorBoard
from os.path import join as join_directory
from tensorflow.config import experimental as tf_gpu


gpus = tf_gpu.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf_gpu.set_memory_growth(gpu, True)
    logical_gpus = tf_gpu.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


optimizer = tf_keras.optimizers.Adam(learning_rate=config.training.learning_rate)
model = Model()
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=metrics)

train_split = config.data.split_prefix.format("train")
validation_split = config.data.split_prefix.format("val")
test_split = config.data.split_prefix.format("test")

log_dir = join_directory(
    config.model.working_dir,
    config.training.log_dir.format(
        config.model.training_label))

ckpt_dir = join_directory(
    config.model.working_dir,
    config.training.ckpt_dir.format(
        config.model.training_label))

def scheduler(epoch, lr):
    if epoch < 20 or epoch > 50:
        pass
    if epoch % 10 == 0:
        lr *= 0.1
    else:
        pass
    
    print(tf.summary.scalar("LearningRate", data=lr))
    return lr
  

ckpt_manager = CheckpointManager(optimizer, model, ckpt_dir)
log_manager = TensorBoard(log_dir=log_dir, update_freq=20)
lr_manager = tf.keras.callbacks.LearningRateScheduler(scheduler)
ckpt_manager.ckpt_manager.restore_or_initialize()

model.fit(
    Dataset(train_split),
    epochs=config.training.epochs,
    callbacks=[ckpt_manager, log_manager, lr_manager],
    validation_data=Dataset(validation_split)
)
