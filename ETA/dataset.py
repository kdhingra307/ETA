#%%
from glob import glob
import tensorflow as tf
import numpy as np
from ETA import config

#%%

mean, std = config.data.mean, config.data.std
mean_expanded = np.array(mean).reshape([1, 1, -1])
std_expanded = np.array(std).reshape([1, 1, -1])

def calculate_random_walk_matrix(adj_mx):
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx)
    return random_walk_mx

def get_data(split_label):

    batch_sampler = sampling()

    def tf_map(file_name):

        data = np.load(file_name)
        x, y = data['x'], data['y'][:, :, 0]

        mask = (y > 0) * 1

        y = (y-mean[0])/std[0]
        x = (x-mean_expanded)/std_expanded

        y = np.stack([y, mask], axis=-1).astype(np.float32)

        return x.astype(np.float32) , y

    files = glob("{}/{}/{}/*.npz".format(config.model.working_dir,
                                         config.data.path_pattern,
                                         split_label))
    tf_dataset = tf.data.Dataset.from_tensor_slices(files)
    tf_dataset = tf_dataset.shuffle(config.data.shuffle)
    tf_dataset = tf_dataset.map(lambda x: tf.numpy_function(tf_map,
                                                            [x],
                                                            [tf.float32,
                                                                tf.float32],
                                                            name="load_each_file"))

    tf_dataset = tf_dataset.map(lambda x, y: (tf.ensure_shape(
        x, [None, config.model.num_nodes, 2]), tf.ensure_shape(y, [None, config.model.num_nodes, 2])))
    
    tf_dataset = tf_dataset.batch(batch_size=config.model.batch_size,
                                  drop_remainder=True)

    def second_map(x, y):
        positions = batch_sampler.sample()

        adj_mx = batch_sampler.adjacency_matrix[positions][:, positions]
        adj_mx = tf.convert_to_tensor(adj_mx, dtype=tf.float32)
        x = tf.gather(x, indices=positions, axis=2)
        y = tf.gather(y, indices=positions, axis=2)

        return adj_mx, x, y

    tf_dataset = tf_dataset.map(second_map)


    tf_dataset = tf_dataset.prefetch(config.data.prefetch)

    return tf_dataset


#%%
class sampling:

    def __init__(self, sampler="random"):

        self.adjacency_matrix = calculate_random_walk_matrix(np.load("{}/{}/adj_matrix.npz".format(
            config.model.working_dir, config.model.static_data_dir))['arr_0'].astype(np.float32))
        
        self.n_init = config.model.graph_batch_size
        self.probab = np.sum(self.adjacency_matrix**2, axis=0)
        self.probab = self.probab/np.sum(self.probab)
    
    def sample(self):
        samples =  np.random.multinomial(1, self.probab, self.n_init)
        positions = np.argmax(samples, axis=-1)

        return positions

# %%
