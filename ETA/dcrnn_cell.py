import tensorflow.keras as tf_keras
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.ops.gen_array_ops import const
from tensorflow.python.ops.gen_math_ops import prod_eager_fallback
from ETA import config


class DCGRUCell(tf.keras.layers.AbstractRNNCell):
    def get_initial_state(self, inputs, batch_size, dtype):

        return tf.zeros(
            [batch_size, tf.shape(inputs)[1], self._num_units], dtype=dtype
        )

    @property
    def output_size(self):
        if self._num_proj:
            return (self._num_nodes, self._num_proj)
        else:
            return (self._num_nodes, self._num_units)

    @property
    def state_size(self):

        return None

    def __init__(
        self,
        num_units,
        adj_mx,
        max_diffusion_step,
        num_nodes,
        num_proj=None,
        activation=tf.nn.tanh,
        reuse=None,
        filter_type="laplacian",
        use_gc_for_ru=True,
    ):

        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        if num_proj != None:
            self.projection_layer = tf_keras.Sequential(
                [
                    tf_keras.layers.Dense(
                        units=64, activation=tf_keras.layers.LeakyReLU(0.2)
                    ),
                    tf_keras.layers.Dense(
                        units=32, activation=tf_keras.layers.LeakyReLU(0.2)
                    ),
                    tf_keras.layers.Dense(units=num_proj),
                ]
            )

    @staticmethod
    def _build_sparse_matrix(L, fac=None):
        if fac is not None:
            return tf.constant(L.todense() / fac)
        else:
            return tf.constant(L.todense())
        # return tf.constant(
        #     [np.arange(207) for _ in range(207)], dtype=tf.float32
        # )
        # L = L.tocoo()
        # indices = np.column_stack((L.row, L.col))
        # L = tf.SparseTensor(indices, L.data, L.shape)
        # return tf.sparse.reorder(L)

    def build(self, inp_shape):

        inpt_features = (inp_shape[-1] + 128) * 4

        kernel_initializer = tf_keras.initializers.GlorotUniform()
        bias_initializer = tf_keras.initializers.Zeros()
        self.w1 = tf.Variable(
            initial_value=kernel_initializer(
                shape=(inpt_features, 2 * self._num_units), dtype=tf.float32
            ),
            trainable=True,
            name="w1",
        )
        self.w2 = tf.Variable(
            initial_value=kernel_initializer(
                shape=(inpt_features, self._num_units), dtype=tf.float32
            ),
            trainable=True,
            name="w2",
        )

        self.b1 = tf.Variable(
            initial_value=bias_initializer(
                shape=(2 * self._num_units,), dtype=tf.float32
            ),
            trainable=True,
            name="b1",
        )
        self.b2 = tf.Variable(
            initial_value=bias_initializer(
                shape=(self._num_units,), dtype=tf.float32
            ),
            trainable=True,
            name="b2",
        )

        self.batch_size = inp_shape[0]

    @tf.function
    def call(self, inputs, state, constants=None, scope=None, training=False):

        """
            inputs_shape [BatchSize, Num_Nodes, Inp_features]
            state_shape [BatchSize, Num_Nodes, Num_units]

        Returns
        -------
        [type]
            [description]
        """
        position = constants[0]

        state = tf.reshape(state, [tf.shape(state[0])[0], -1, self._num_units])
        num_nodes = tf.shape(state)[1]

        output_size = 2 * self._num_units

        value = tf.sigmoid(
            self._gconv(
                inputs,
                state,
                output_size,
                bias_start=1.0,
                _supports=position,
                training=training,
            )
        )

        value = tf.reshape(value, (-1, num_nodes, output_size))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)

        c = self._gconv(
            inputs,
            r * state,
            self._num_units,
            _supports=position,
            training=training,
        )

        if self._activation is not None:
            c = self._activation(c)

        output = new_state = u * state + (1 - u) * c

        if self._num_proj is not None:
            output = self.projection_layer(output, training=training)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    @tf.function
    def _gconv(
        self,
        inputs,
        state,
        output_size,
        _supports,
        bias_start=0.0,
        training=False,
    ):

        inputs_and_state = tf.concat([inputs, state], axis=2)
        num_nodes = tf.shape(inputs)[1]
        num_inpt_features = inputs_and_state.shape[-1]

        x0 = tf.reshape(
            tf.transpose(inputs_and_state, [1, 0, 2]),
            [num_nodes, -1],
        )
        output = []

        for i in range(2):

            x1 = tf.matmul(_supports[i], x0)
            output.append(x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * tf.matmul(_supports[i], x1) - x0
                output.append(x2)
                x1, x0 = x2, x1

        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(
            tf.stack(output, axis=-1),
            [num_nodes, batch_size, num_inpt_features, -1],
        )

        x = tf.transpose(x, [1, 0, 3, 2])
        x = tf.reshape(x, [batch_size, num_nodes, -1])

        if output_size == self._num_units:
            x = tf.matmul(x, self.w2) + self.b2
        else:
            x = tf.matmul(x, self.w1) + self.b1

        return x


class DCGRUBlock(tf_keras.layers.Layer):
    def __init__(self, dcrnn_cells, num_nodes, steps_to_predict, encode=True):
        super(DCGRUBlock, self).__init__()

        self.is_encoder = encode
        self.cells = dcrnn_cells
        self.num_nodes = num_nodes
        self.steps_to_predict = steps_to_predict
        # self.counter = config.model.counter_position
        if encode:
            self.block = tf.keras.layers.RNN(self.cells, return_state=True)

    def build(self, x_shape):
        self.batch_size = x_shape[0]

    def encode(self, x, pos, training=False):
        state = self.block(
            x,
            constants=[pos],
            initial_state=(
                tf.zeros([tf.shape(x)[0], tf.shape(x)[2], 128]),
                tf.zeros([tf.shape(x)[0], tf.shape(x)[2], 128]),
            ),
            training=training,
        )
        return state[1:]

    @tf.function
    def decay_teacher_coefficient(self):
        decay_rate = config.model.teacher_decay_rate

        teacher_coeff = decay_rate / (
            decay_rate + tf.exp(self.counter / decay_rate)
        )
        tf.summary.scalar(name="teacher_decay_coefficient", data=teacher_coeff)
        self.counter += 1

        return teacher_coeff

    @tf.function
    def decode(self, state, pos=None, x_targ=None, training=False):

        init = tf.zeros(
            [tf.shape(state[0])[0], tf.shape(state[0])[1], 1], dtype=tf.float32
        )

        state = tuple(state)

        to_return = tf.TensorArray(
            size=self.steps_to_predict, dtype=tf.float32
        )
        for i in range(self.steps_to_predict):
            init, state = self.cells(
                init, states=state, constants=[pos], training=training
            )
            to_return = to_return.write(i, init)
            init = tf.stop_gradient(init)

        return tf.transpose(tf.squeeze(to_return.stack(), axis=-1), [1, 0, 2])

    def call(self, x, state, pos, training=False):
        if self.is_encoder:
            return self.encode(x, pos=pos, training=training)
        else:
            return self.decode(
                state=state, x_targ=x, pos=pos, training=training
            )
