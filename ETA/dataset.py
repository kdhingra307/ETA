#%%
from glob import glob
import tensorflow as tf
import numpy as np
from ETA import config
from scipy.sparse.linalg.eigen.arpack import eigsh

#%%

mean, std = config.data.mean, config.data.std
mean_expanded = np.array(mean).reshape([1, 1, -1])
std_expanded = np.array(std).reshape([1, 1, -1])


def calculate_random_walk_matrix(adj_mx):
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = np.diag(d_inv)

    return adj_mx.dot(d_mat_inv).T.dot(d_mat_inv)


def get_data(split_label):

    # batch_sampler = sampling()

    def tf_map(file_name):

        data = np.load(file_name)
        x, y = data["x"], data["y"][:, :, 0]

        mask = (y > 0) * 1

        y = (y - mean[0]) / std[0]
        x = (x - mean_expanded) / std_expanded

        y = np.stack([y, mask], axis=-1).astype(np.float32)

        return x.astype(np.float32), y

    files = sorted(
        glob(
            "{}/{}/{}/*.npz".format(
                config.model.working_dir, config.data.path_pattern, split_label
            )
        )
    )
    tf_dataset = tf.data.Dataset.from_tensor_slices(files)
    tf_dataset = tf_dataset.shuffle(config.data.shuffle, seed=1234)
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

    tf_dataset = tf_dataset.batch(
        batch_size=config.model.batch_size,
    )

    def second_map(x, y):
        # positions = batch_sampler.sample()

        adj_mx = batch_sampler.adjacency_matrix
        # norm = batch_sampler.probab_individ[positions][:, positions]

        # adj_mx /= norm
        # adj_mx *= batch_sampler.probab[positions].reshape([1, -1])

        adj_mx = tf.convert_to_tensor(adj_mx, dtype=tf.float32)
        # x = tf.gather(x, indices=positions, axis=2)
        # y = tf.gather(y, indices=positions, axis=2)

        return adj_mx, x, y

    # tf_dataset = tf_dataset.map(second_map)
    tf_dataset = tf_dataset.cache(
        "{}/cache_{}".format(config.model.working_dir, split_label)
    )

    tf_dataset = tf_dataset.prefetch(config.data.prefetch)

    return tf_dataset


#%%
class sampling:
    def __init__(self, sampler="random"):

        # adjacency_matrix = [np.eye(len(mat))]
        # for e in range(3):
        #     adjacency_matrix.append(adjacency_matrix[-1].dot(nmat))

        # adjacency_matrix1 = calculate_random_walk_matrix(mat.T).T
        # support = []
        # support.append(adjacency_matrix)
        # support.append(adjacency_matrix1)

        # support.append(
        #     2 * adjacency_matrix.dot(adjacency_matrix)
        #     - np.eye(len(adjacency_matrix))
        # )
        # support.append(
        #     2 * adjacency_matrix1.dot(adjacency_matrix1)
        #     - np.eye(len(adjacency_matrix))
        # )

        # self.adjacency_matrix = np.stack(adjacency_matrix, axis=-1)
        # self.adjacency_matrix = chebyshev_polynomials(adjacency_matrix, 3)
        # print(self.adjacency_matrix.shape)

        self.n_init = config.model.graph_batch_size
        self.probab_individ = self.adjacency_matrix ** 2
        self.probab = np.sum(self.probab_individ, axis=-1)
        self.probab = self.probab / np.sum(self.probab)

    def sample(self):
        return np.arange(self.n_init)
        samples = np.random.multinomial(1, self.probab, self.n_init)
        positions = np.argmax(samples, axis=-1)

        return positions


# %%
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    # laplacian = np.eye(adj.shape[0]) - adj
    # largest_eigval, _ = eigsh(laplacian, 1, which="LM")
    # scaled_laplacian = (2.0 / largest_eigval[0]) * laplacian - np.eye(
    #     adj.shape[0]
    # )

    t_k = list()
    t_k.append(np.eye(adj.shape[0]))
    t_k.append(adj)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two):
        return 2 * adj.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2]))

    return np.stack(t_k, axis=-1)
