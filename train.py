from ETA import Model, Dataset, loss_function, metrics, config
from datetime import datetime
import tensorflow.keras as tf_keras
from tensorflow.keras.callbacks import TensorBoard
from os.path import join as join_directory

model = Model()
model.compile(optimizer="adam",
              loss=loss_function,
              metrics=metrics)

train_split = config.data.split_prefix.format("train")
validation_split = config.data.split_prefix.format("val")
test_split = config.data.split_prefix.format("test")

training_label = "lstm_with_conv1d"
temporal_label = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = config.training.log_dir.format(training_label, temporal_label)

model.fit(
    Dataset(train_split),
    epochs=10,
    callbacks=[TensorBoard(log_dir=join_directory(config.model.working_dir,log_dir))],
    validation_data=Dataset(validation_split)
)
