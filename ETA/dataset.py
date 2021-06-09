#%%
from glob import glob
import tensorflow as tf
import numpy as np
from ETA import config

#%%

mean, std = config.data.mean, config.data.std
mean_expanded = np.array(mean).reshape([1, 1, -1])
std_expanded = np.array(std).reshape([1, 1, -1])


def get_data(split_label):

    batch_sampler = sampling()

    def tf_map(file_name):

        data = np.load(file_name)
        x, y = data["x"], data["y"][:, :, 0]

        mask = (y > 0) * 1

        y = (y - mean[0]) / std[0]
        x = (x - mean_expanded) / std_expanded

        y = np.stack([y, mask], axis=-1).astype(np.float32)

        return x.astype(np.float32), y

    files = glob(
        "{}/{}/{}/*.npz".format(
            config.model.working_dir, config.data.path_pattern, split_label
        )
    )
    tf_dataset = tf.data.Dataset.from_tensor_slices(files)
    tf_dataset = tf_dataset.shuffle(config.data.shuffle)
    tf_dataset = tf_dataset.map(
        lambda x: tf.numpy_function(
            tf_map, [x], [tf.float32, tf.float32], name="load_each_file"
        )
    )

    tf_dataset = tf_dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, [None, config.model.num_nodes, 2]),
            tf.ensure_shape(y, [None, config.model.num_nodes, 2]),
        )
    )

    tf_dataset = tf_dataset.batch(batch_size=config.model.batch_size)

    def second_map(x, y):
        positions = batch_sampler.sample()

        x = tf.gather(x, indices=positions, axis=2)
        y = tf.gather(y, indices=positions, axis=2)

        return positions, x, y

    tf_dataset = tf_dataset.map(second_map)

    tf_dataset = tf_dataset.prefetch(config.data.prefetch)

    return tf_dataset


#%%
class sampling:
    def __init__(self, sampler="random"):

        adj = np.load(
            "{}/{}/metr_adj_matrix.npz".format(
                config.model.working_dir, config.model.static_data_dir
            )
        )["arr_0"].astype(np.float32)

        self.n_init = config.model.graph_batch_size
        self.probab_individ = adj ** 2
        self.probab = np.sum(self.probab_individ, axis=0)
        self.probab = self.probab / np.sum(self.probab)

    def sample(self):

        samples = tf.random.categorical(
            tf.math.log(self.probab[None, :]), self.n_init
        )[0]
        return samples
        # return tf.random.shuffle(np.arange(207))
