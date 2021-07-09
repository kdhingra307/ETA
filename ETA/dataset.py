from glob import glob
import tensorflow as tf
import numpy as np
from ETA import config

#%%

mean, std = config.data.mean, config.data.std
mean_expanded = np.array(mean).reshape([1, 1, -1])
std_expanded = np.array(std).reshape([1, 1, -1])

adj_mx = np.load(
    "{}/{}/spearson_custom.npz".format(
        config.model.working_dir, config.model.static_data_dir
    )
)["arr_0"].astype(np.float32)

non_zero_rows = np.load(
    "/home/pravesh/speech_work/ETA/models/ETA/data/static/custom_non_zero_1165.npy"
)


def calculate_random_walk_matrix(adj_mx):
    d = tf.reduce_sum(adj_mx, axis=1)
    d_inv = tf.math.pow(d, -0.5)
    d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)
    d_mat_inv = tf.linalg.diag(d_inv)
    return tf.matmul(
        tf.transpose(tf.matmul(adj_mx, d_mat_inv), [1, 0]), d_mat_inv
    )


base_supports = [
    tf.constant(adj_mx, dtype=tf.float32),
]


def get_data(split_label):

    batch_sampler = rwt_sampling()

    def tf_map(file_name):

        data = np.load(file_name)
        x, y = (
            np.transpose(data["x"], [1, 0, 2])[:, non_zero_rows],
            np.transpose(data["y"], [1, 0, 2])[:, non_zero_rows, 0],
        )

        x = (x - mean_expanded) / std_expanded
        mask = (y > 0) * 1

        y = (y - mean[0]) / std[0]

        y = np.stack([y, mask], axis=-1).astype(np.float32)

        return x.astype(np.float32), y

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
            [tf.float32, tf.float32],
            name="load_each_file",
        )
    )

    tf_dataset = tf_dataset.batch(batch_size=config.model.batch_size)

    def second_map(x, y):
        positions = batch_sampler.sampler[split_label]()

        x = tf.gather(x, indices=positions, axis=2)
        y = tf.gather(y, indices=positions, axis=2)

        return x, y

    tf_dataset = tf_dataset.map(second_map)

    tf_dataset = tf_dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, [None, None, 256, 2]),
            tf.ensure_shape(y, [None, None, 256, 2]),
        )
    )
    # tf_dataset = tf_dataset.map(
    #     lambda x, y: (
    #         tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, 12, 2]),
    #         tf.reshape(tf.transpose(y, [0, 2, 1, 3]), [-1, 12, 2]),
    #     )
    # )

    tf_dataset = tf_dataset.prefetch(config.data.prefetch)

    return tf_dataset


#%%
class node_sampling:
    def __init__(self, sampler="random"):

        adj = np.load(
            "{}/{}/metr_adj_matrix.npz".format(
                config.model.working_dir, config.model.static_data_dir
            )
        )["arr_0"].astype(np.float32)

        self.n_init = config.model.graph_batch_size
        self.probab_individ = adj ** 2
        self.probab = np.sum(self.probab_individ, axis=-1)
        self.probab = self.probab / np.sum(self.probab)

        self.sampler = {
            "melr_train": self.sample,
            "melr_val": lambda: tf.range(256),
        }

    def sample(self):

        samples = tf.random.categorical(
            tf.math.log(self.probab[None, :]), self.n_init
        )[0]
        samples = tf.unique(samples)[0]
        return samples
        # return tf.random.shuffle(np.arange(207))


class rwt_sampling:
    def __init__(self, sampler="random"):

        self.adj = (
            np.load(
                "{}/{}/spearson_custom.npz".format(
                    config.model.working_dir, config.model.static_data_dir
                )
            )["arr_0"].astype(np.float32)
            > 0
        )

        self.n_init = config.model.graph_batch_size
        self.n_nodes = config.model.num_nodes

        self.sampler = {
            "custom_new_train": self.sample,
            "custom_new_val": self.sample,
            "custom_new_test": self.sample,
        }

    def dummy(self):
        nodes = np.array([np.random.randint(self.n_nodes)])

        while len(nodes) < self.n_init:
            cur_nodes = len(nodes)
            neighbours = np.array([], dtype=np.int32)
            for e in nodes:
                neighbours = np.union1d(neighbours, np.nonzero(self.adj[e])[0])

            chosen_neighbours = (
                neighbours
                if len(neighbours) < 12
                else np.random.choice(neighbours, 12, replace=False)
            )
            nodes = np.union1d(
                nodes,
                chosen_neighbours,
            )

            if cur_nodes == len(nodes):
                remaining_nodes = np.delete(np.arange(self.n_nodes), nodes)
                nodes = np.concatenate(
                    [nodes, np.random.choice(remaining_nodes, 1)]
                )

        return np.array(nodes)[: self.n_init].astype(np.int32)

    def sample(self):

        output = tf.numpy_function(self.dummy, [], tf.int32)

        return tf.ensure_shape(output, [self.n_init])
