from glob import glob
import tensorflow as tf
import numpy as np
from ETA import config


def get_data(split_label):

    def tf_map(file_name):

        data = np.load(file_name)
        x, y = data['x'], data['y'][:, :, 0]

        eta_data = x[:, :, 0]
        time_info = x[:, :1, 1]
        mask = (y > 0) * 1

        x = np.concatenate([eta_data, time_info], axis=1).astype(np.float32)
        y = np.stack([y, mask], axis=-1).astype(np.float32)

        return x, y

    files = glob("{}/{}/{}/*.npz".format(config.model.working_dir,
                                         config.data.path_pattern,
                                         split_label))

    tf_dataset = tf.data.Dataset.from_tensor_slices(files)
    tf_dataset = tf_dataset.shuffle(config.data.shuffle)
    tf_dataset = tf_dataset.map(lambda x: tf.numpy_function(tf_map,
                                                            [x],
                                                            [tf.float32, tf.float32],
                                                            name="load_each_file"))
    tf_dataset = tf_dataset.cache(
        "{}/cache".format(config.model.working_dir))
    tf_dataset = tf_dataset.batch(batch_size=config.model.batch_size,
                                  drop_remainder=True)

    tf_dataset = tf_dataset.map(lambda x, y: (tf.ensure_shape(x, [None, None, 6119]), tf.ensure_shape(y, [None, None, 6118, 2])))

    tf_dataset = tf_dataset.prefetch(config.data.prefetch)

    return tf_dataset
