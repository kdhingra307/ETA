from glob import glob
import tensorflow as tf
import numpy as np
from ETA import config


mean, std = config.data.mean, config.data.std
mean_expanded = np.array(mean).reshape([1, 1, -1])
std_expanded = np.array(std).reshape([1, 1, -1])


def get_data(split_label):
    def tf_map(file_name):

        data = np.load(file_name)
        x, y = data["x"], data["y"][:, :, 0]

        x_mask = (x[:, :, 0] > 0).astype(np.float32)
        x_mask *= np.random.uniform(size=x_mask.shape)
        x_mask = x_mask > 0.2

        x = (x - mean_expanded) / std_expanded

        x_mask = np.concatenate(
            [x_mask, np.ones(x_mask.shape[:1])[:, None]], axis=-1
        )

        x = np.concatenate([x[:, :, 0] * x_mask, x[:, :1, 1]], axis=1).astype(
            np.float32
        )
        x1 = [np.zeros(len(x[0]))]
        dt = [np.zeros(len(x[0]))]
        for e in range(len(x)):
            x1.append((x1[-1] * (1 - x_mask[e])) + x_mask[e] * x[e])
            dt.append(((1 + dt[-1]) * (1 - x_mask[e])) + x_mask[e])

        x1 = np.stack(x1[1:], axis=0)
        dt = np.stack(dt[1:], axis=0) / 12

        x2 = np.sum(x_mask * x, axis=0) / (np.sum(x_mask, axis=0) + 1e-12)

        x = np.concatenate([x, x1, x_mask, dt], axis=-1)

        y_mask = (y > 0) * 1
        y = (y - mean[0]) / std[0]
        y = np.stack([y, y_mask], axis=-1).astype(np.float32)

        return (
            x.astype(np.float32),
            y.astype(np.float32),
            x2.astype(np.float32),
        )

    files = glob(
        "{}/{}/{}/*.npz".format(
            config.model.working_dir, config.data.path_pattern, split_label
        )
    )

    tf_dataset = tf.data.Dataset.from_tensor_slices(files)

    tf_dataset = tf_dataset.shuffle(config.data.shuffle, seed=1234)
    tf_dataset = tf_dataset.map(
        lambda x: tf.numpy_function(
            tf_map,
            [x],
            [tf.float32, tf.float32, tf.float32],
            name="load_each_file",
        )
    )

    # tf_dataset = tf_dataset.cache(
    #     "{}/cache_{}".format(config.model.working_dir, split_label)
    # )
    tf_dataset = tf_dataset.batch(
        batch_size=config.model.batch_size,
    )

    tf_dataset = tf_dataset.map(
        lambda x, y, z: (
            tf.ensure_shape(x, [None, None, (config.model.num_nodes + 1) * 4]),
            tf.ensure_shape(y, [None, None, config.model.num_nodes, 2]),
            tf.ensure_shape(z, [None, config.model.num_nodes + 1]),
        )
    )

    tf_dataset = tf_dataset.prefetch(config.data.prefetch)

    return tf_dataset
