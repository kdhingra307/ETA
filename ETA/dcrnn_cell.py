import tensorflow.keras as tf_keras
import numpy as np
import tensorflow as tf
import scipy.sparse as sp


class DCGRUCell(tf.keras.layers.AbstractRNNCell):


    def get_initial_state(self, inputs, batch_size, dtype):
        return tf.zeros([batch_size, self._num_nodes, self._num_units], dtype=dtype)
    
    @property
    def output_size(self):
        if self._num_proj:
            return (self._num_nodes, self._num_proj)
        else:
            return (self._num_nodes, self._num_units)
    

    @property
    def state_size(self):
        
        return (self._num_nodes*self._num_units)

    

    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian", use_gc_for_ru=True):

        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        
        supports = []
        supports.append(calculate_random_walk_matrix(adj_mx).T)
        supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))
        
        if num_proj != None:
            self.projection_layer = tf_keras.layers.Dense(units=num_proj)

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse.reorder(L)


    def build(self, inp_shape):

        inpt_features = (inp_shape[-1] + 64) * 4
        
        kernel_initializer = tf_keras.initializers.GlorotUniform()
        bias_initializer = tf_keras.initializers.Zeros()
        self.w1 = tf.Variable(initial_value=kernel_initializer(shape=(inpt_features, 2*self._num_units), dtype=tf.float32), trainable=True)
        self.w2 = tf.Variable(initial_value=kernel_initializer(shape=(inpt_features, self._num_units), dtype=tf.float32), trainable=True)

        self.b1 = tf.Variable(initial_value=bias_initializer(shape=(2*self._num_units,), dtype=tf.float32), trainable=True)
        self.b2 = tf.Variable(initial_value=bias_initializer(shape=(self._num_units,), dtype=tf.float32), trainable=True)


        # self.gconv_layer1 = tf_keras.layers.Dense(units=2*self._num_units, kernel_initializer=tf_keras.initializers.Constant(w1), bias_initializer=tf_keras.initializers.Constant(b1))
        # self.gconv_layer2 = tf_keras.layers.Dense(units=self._num_units, kernel_initializer=tf_keras.initializers.Constant(w2), bias_initializer=tf_keras.initializers.Constant(b2))


        self.batch_size = inp_shape[0]

    @tf.function
    def call(self, inputs, state, scope=None):

        """
            inputs_shape [BatchSize, Num_Nodes, Inp_features]
            state_shape [BatchSize, Num_Nodes, Num_units]

        Returns
        -------
        [type]
            [description]
        """

        state = tf.reshape(state, [-1, self._num_nodes, self._num_units])
        
        output_size = 2 * self._num_units
        value = tf.sigmoid(self._gconv(inputs, state, output_size, bias_start=1.0))
        value = tf.reshape(value, (-1, self._num_nodes, output_size))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)

        c = self._gconv(inputs, r * state, self._num_units)

        if self._activation is not None:
            c = self._activation(c)
        
        output = new_state = u * state + (1 - u) * c

        if self._num_proj is not None:
            output = self.projection_layer(output)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    @tf.function
    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        

        inputs_and_state = tf.concat([inputs, state], axis=2)
        num_inpt_features = inputs_and_state.shape[-1]

        x = inputs_and_state
        x0 = tf.reshape(tf.transpose(x, perm=[1, 2, 0]), [self._num_nodes, -1])
        output = []

        for support in self._supports:
            x1 = tf.sparse.sparse_dense_matmul(support, x0)
            output.append(x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * tf.sparse.sparse_dense_matmul(support, x1) - x0
                output.append(x2)
                x1, x0 = x2, x1

        x = tf.reshape(tf.concat(output, axis=-1), [self._num_nodes, num_inpt_features, self.batch_size, -1])
        x = tf.transpose(x, [2, 0, 1, 3])
        x = tf.reshape(x, [self.batch_size, self._num_nodes, -1])

        if output_size == self._num_units:
            x = tf.matmul(x, self.w2) + self.b2
            # x = self.gconv_layer2(x)
        else:
            x = tf.matmul(x, self.w1) + self.b1
            # x = self.gconv_layer1(x)
        
        return x

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

class DCGRUBlock(tf_keras.layers.Layer):

    def __init__(self, dcrnn_cells, num_nodes, steps_to_predict, encode=True):
        super(DCGRUBlock, self).__init__()

        self.is_encoder = encode
        self.cells = dcrnn_cells
        self.num_nodes = num_nodes
        self.steps_to_predict = steps_to_predict
        if encode:
            self.block = tf.keras.layers.RNN(self.cells, return_state=True)
        
    def build(self, x_shape):
        self.batch_size = x_shape[0]
    
    def encode(self, x):
        state = self.block(x)
        return state[-1]
    
    @tf.function
    def decode(self, state, x_targ=None):
        
        init = tf.zeros([self.batch_size, self.num_nodes, 1], dtype=tf.float32)
        nstate = self.cells.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        state = [state, nstate[1]]

        to_return = []
        if x_targ is None:
            for i in range(self.steps_to_predict):
                init, state = self.cells(init, states=state)
                to_return.append(init)
            return tf.stack(to_return, axis=1)
        else:
            for i in range(self.steps_to_predict):
                output, state = self.cells(init, states=state)
                to_return.append(output)

                if np.random.rand() > 0.5:
                    init = output
                else:
                    init = x_targ[:, i]

            return tf.stack(to_return, axis=1)
        
    
    def call(self, x, state):
        if self.is_encoder:
            return self.encode(x)
        else:
            return self.decode(state, x)
            



            




