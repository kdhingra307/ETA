import tensorflow as tf
from ETA.utils import get_config, mse, mae, mape, rmse, CheckpointManager

config = get_config("./config.yaml")
metrics = {
    "seq2seq/ar": [mse, mae, mape, rmse],
    "gan/ar": tf.keras.metrics.BinaryAccuracy(),
    "seq2seq/ttf": [mse, mae, mape, rmse],
    "gan/ttf": tf.keras.metrics.BinaryAccuracy(),
}

loss_function = mse


from ETA.dataset import get_data as Dataset
from ETA.model import Model
