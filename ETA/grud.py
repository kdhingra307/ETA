import tensorflow.keras as tf_keras
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.ops.gen_array_ops import const
from tensorflow.python.ops.gen_math_ops import prod_eager_fallback
from ETA import config


class GRUDCell(tf.keras.layers.AbstractRNNCell):
    def get_initial_state(self, inputs, batch_size, dtype):

        return tf.zeros(
            [batch_size, self._num_nodes, self._num_units], dtype=dtype
        )

    @property
    def output_size(self):
        if self._num_proj:
            return (self._num_nodes, self._num_proj)
        else:
            return (self._num_nodes, self._num_units)

    @property
    def state_size(self):
        if self._num_proj:
            return self._num_nodes * self._num_proj
        else:
            return self._num_nodes * self._num_units

    def __init__(
        self,
        num_units,
        num_nodes,
        num_proj=None,
        activation=tf.nn.tanh,
    ):

        super(GRUDCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units

        self.first_layer = GSConv(units=num_units * 2, should=True)
        self.second_layer = GSConv(num_units, should=True)

        self.x_prev = tf.keras.layers.Dense(2, name="x_prev")
        self.x_prev1 = [
            tf.keras.layers.Dense(
                16,
                name="x_prev",
                activation=tf_keras.layers.LeakyReLU(alpha=0.2),
            )
            for _ in range(4)
        ]
        self.norms = [tf.keras.layers.BatchNormalization() for _ in range(3)]
        self.h_prev = tf.keras.layers.Dense(num_units, name="h_prev")

        if num_proj != None:
            self.projection_layer = tf_keras.Sequential(
                [
                    tf_keras.layers.Dense(
                        units=64, activation=tf_keras.layers.LeakyReLU(0.2)
                    ),
                    tf_keras.layers.BatchNormalization(),
                    tf_keras.layers.Dense(
                        units=32, activation=tf_keras.layers.LeakyReLU(0.2)
                    ),
                    tf_keras.layers.Dense(units=1),
                ]
            )

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
        support = constants[0]
        x2 = constants[1]

        x, x1, mask, dt = tf.split(inputs, num_or_size_splits=4, axis=-1)

        x_l = self.x_prev1[0](dt)
        for e in range(1, 4):
            x_l = self.norms[e - 1](x_l)
            x_l += self.x_prev1[e](x_l)

        x_prev_mask = tf.exp(
            -1 * tf.clip_by_value(self.x_prev(x_l), 0, tf.float32.max)
        )

        inputs = (x * mask) + (
            (1 - mask) * (x_prev_mask * x1 + (1 - x_prev_mask) * x2)
        )

        state = tf.reshape(state, [tf.shape(state[0])[0], -1, self._num_units])

        h_prev_mask = tf.exp(
            -1 * tf.clip_by_value(self.h_prev(dt), 0, tf.float32.max)
        )
        state = h_prev_mask * state

        inputs_and_state = tf.concat([inputs, state], axis=2)

        value = tf.sigmoid(
            self.first_layer(inputs_and_state, support, training=training)
        )

        r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)

        inputs_and_state = tf.concat([inputs, r * state], axis=2)
        c = self.second_layer(inputs_and_state, support, training=training)

        if self._activation is not None:
            c = self._activation(c)

        output = new_state = u * state + (1 - u) * c

        return output, new_state


class GSConv(tf_keras.layers.Layer):
    def __init__(self, units, should=False):
        super(GSConv, self).__init__()

        self._hidden = units // 2

        self.layer = tf.keras.layers.Dense(
            units=units // 2,
            activation=tf.keras.layers.LeakyReLU(0.2),
        )
        self.layer1 = tf.keras.layers.Dense(
            units=units // 2,
            activation=tf.keras.layers.LeakyReLU(0.2),
        )
        self.batch_norm = tf.keras.layers.Dropout(0.1)

        self.layer2 = tf.keras.layers.Dense(units=units)
        self.should = should

    def call(self, x0, support, training=False):

        output = []
        x = tf.tensordot(support[0], x0, axes=[1, 1])
        x = tf.transpose(x, [1, 0, 2])
        x = self.layer(x)
        output.append(x)

        x = tf.tensordot(support[1], tf.concat([x, x0], axis=-1), axes=[1, 1])
        x = tf.transpose(x, [1, 0, 2])

        x = self.layer1(x)
        x = self.batch_norm(x, training=training)
        output.append(x)

        return self.layer2(tf.concat(output, axis=-1))
