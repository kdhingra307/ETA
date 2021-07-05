import tensorflow.keras as tf_keras
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.ops.gen_array_ops import const
from tensorflow.python.ops.gen_math_ops import prod_eager_fallback
from ETA import config


class GRUDCell(tf.keras.layers.AbstractRNNCell):
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

        super(GRUDCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        self._supports = [
            tf.constant(adj_mx, dtype=tf.float32),
            tf.constant(adj_mx.T, dtype=tf.float32),
        ]

        self.x_prev = tf.keras.layers.Dense(2, name="x_prev")
        self.h_prev = tf.keras.layers.Dense(num_units, name="h_prev")

        if num_proj != None:
            self.projection_layer = tf_keras.Sequential(
                [
                    tf_keras.layers.Dense(
                        units=32, activation=tf_keras.layers.LeakyReLU(0.2)
                    ),
                    tf_keras.layers.BatchNormalization(),
                    tf_keras.layers.Dense(
                        units=16, activation=tf_keras.layers.LeakyReLU(0.2)
                    ),
                    tf_keras.layers.Dense(units=num_proj),
                ]
            )

    def build(self, inp_shape):

        inpt_features = (inp_shape[-1] + 64) * 4

        kernel_initializer = tf_keras.initializers.GlorotUniform()
        bias_initializer = tf_keras.initializers.Zeros()
        self.w1 = tf.Variable(
            initial_value=kernel_initializer(
                shape=(inpt_features, 2 * self._num_units), dtype=tf.float32
            ),
            trainable=True,
        )
        self.w2 = tf.Variable(
            initial_value=kernel_initializer(
                shape=(inpt_features, self._num_units), dtype=tf.float32
            ),
            trainable=True,
        )

        self.b1 = tf.Variable(
            initial_value=bias_initializer(
                shape=(2 * self._num_units,), dtype=tf.float32
            ),
            trainable=True,
        )
        self.b2 = tf.Variable(
            initial_value=bias_initializer(
                shape=(self._num_units,), dtype=tf.float32
            ),
            trainable=True,
        )
        self.built = True

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
        x2 = constants[1]

        x, x1, mask, dt = tf.split(inputs, num_or_size_splits=4, axis=-1)

        x_prev_mask = tf.exp(
            -1 * tf.clip_by_value(self.x_prev(dt), 0, tf.float32.max)
        )

        inputs = (x * mask) + (
            (1 - mask) * (x_prev_mask * x1 + (1 - x_prev_mask) * x2)
        )

        state = tf.reshape(state, [tf.shape(state[0])[0], -1, self._num_units])

        h_prev_mask = tf.exp(
            -1 * tf.clip_by_value(self.h_prev(dt), 0, tf.float32.max)
        )
        state = h_prev_mask * state

        num_nodes = tf.shape(state)[1]

        output_size = 2 * self._num_units
        value = tf.sigmoid(
            self._gconv(
                inputs,
                state,
                output_size,
                bias_start=1.0,
                pos=position,
                training=training,
            )
        )
        value = tf.reshape(value, (-1, num_nodes, output_size))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)

        c = self._gconv(
            inputs,
            r * state,
            self._num_units,
            pos=position,
            training=training,
        )

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
    def _gconv(
        self,
        inputs,
        state,
        output_size,
        pos,
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

        for support in self._supports:

            cur_support = tf.gather(
                tf.gather(support, pos, axis=1), pos, axis=0
            )
            cur_support = calculate_random_walk_matrix(cur_support)

            x1 = tf.matmul(cur_support, x0)
            output.append(x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * tf.matmul(cur_support, x1) - x0
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


def calculate_random_walk_matrix(adj_mx):
    d = tf.reduce_sum(adj_mx, axis=1)
    d_inv = tf.math.pow(d, -1)
    d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)
    d_mat_inv = tf.linalg.diag(d_inv)

    random_walk_mx = tf.matmul(d_mat_inv, adj_mx)

    return random_walk_mx
