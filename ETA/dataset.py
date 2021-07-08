from glob import glob
import tensorflow as tf
import numpy as np
from ETA import config


mean, std = config.data.mean, config.data.std
mean_expanded = np.array(mean).reshape([1, 1, -1])
std_expanded = np.array(std).reshape([1, 1, -1])

non_zero_rows = np.load("./data/static/custom_non_zero_1165.npy")


def get_data(split_label):
    def tf_map(file_name):

        data = np.load(file_name)
        x, y = (
            np.transpose(data["x"], [1, 0, 2])[:, non_zero_rows],
            np.transpose(data["y"], [1, 0, 2])[:, non_zero_rows, 0],
        )

        mask = (y > 0) * 1

        y = (y - mean[0]) / std[0]
        x = (x - mean_expanded) / std_expanded

        y = np.stack([y, mask], axis=-1).astype(np.float32)

        return x.astype(np.float32), y

    files = glob(
        "{}/{}/{}/*npz".format(
            config.model.working_dir, config.data.path_pattern, split_label
        )
    )

    tf_dataset = tf.data.Dataset.from_tensor_slices(files)

    tf_dataset = tf_dataset.shuffle(config.data.shuffle, seed=1234)
    tf_dataset = tf_dataset.map(
        lambda x: tf.numpy_function(
            tf_map, [x], [tf.float32, tf.float32], name="load_each_file"
        )
    )

    tf_dataset = tf_dataset.batch(
        batch_size=config.model.batch_size,
    )

    tf_dataset = tf_dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, [None, None, config.model.num_nodes, 2]),
            tf.ensure_shape(y, [None, None, config.model.num_nodes, 2]),
        )
    )
    tf_dataset = tf_dataset.map(
        lambda x, y: (
            tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, 12, 2]),
            tf.reshape(tf.transpose(y, [0, 2, 1, 3]), [-1, 12, 2]),
        )
    )

    tf_dataset = tf_dataset.prefetch(config.data.prefetch)

    return tf_dataset
