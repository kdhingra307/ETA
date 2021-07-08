import tensorflow.keras as tf_keras
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.ops.gen_array_ops import const
from tensorflow.python.ops.gen_math_ops import prod_eager_fallback
from ETA import config


class GRUCell(tf.keras.layers.AbstractRNNCell):
    def get_initial_state(self, inputs, batch_size, dtype):

        return tf.zeros([batch_size, self._num_units], dtype=dtype)

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __init__(self, units, activation=tf.nn.tanh):

        super(GRUCell, self).__init__()
        self._activation = activation
        self._num_units = units

    def build(self, inp_shape):

        inpt_features = inp_shape[-1] + 128

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
    def call(self, inputs, state, training=False):

        """
            inputs_shape [BatchSize, Num_Nodes, Inp_features]
            state_shape [BatchSize, Num_Nodes, Num_units]

        Returns
        -------
        [type]
            [description]
        """

        state = tf.reshape(state, [-1, self._num_units])

        inputs_and_state = tf.concat([inputs, state], axis=1)
        value = tf.sigmoid(tf.matmul(inputs_and_state, self.w1) + self.b1)

        r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
        inputs_and_state = tf.concat([inputs * r, state], axis=1)

        c = tf.matmul(inputs_and_state, self.w2) + self.b2

        if self._activation is not None:
            c = self._activation(c)

        output = new_state = u * state + (1 - u) * c

        return output, new_state
