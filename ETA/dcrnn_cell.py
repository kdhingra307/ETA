import tensorflow.keras as tf_keras
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from ETA import config


class DCGRUCell(tf.keras.layers.AbstractRNNCell):
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

        return self._num_nodes * self._num_units

    def __init__(
        self,
        num_units,
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
        self._use_gc_for_ru = use_gc_for_ru

        if num_proj != None:
            self.projection_layer = tf_keras.Sequential(
                [
                    tf_keras.layers.Dense(
                        units=128,
                        activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                    ),
                    tf_keras.layers.BatchNormalization(),
                    tf_keras.layers.Dense(
                        units=32,
                        activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                    ),
                    tf_keras.layers.BatchNormalization(),
                    tf_keras.layers.Dense(
                        units=1,
                    ),
                ]
            )

    def build(self, inp_shape):

        inpt_features = inp_shape[-1] + self._num_units

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

    @tf.function
    def call(self, inputs, state, constants, scope=None):

        """
            inputs_shape [BatchSize, Num_Nodes, Inp_features]
            state_shape [BatchSize, Num_Nodes, Num_units]

        Returns
        -------
        [type]
            [description]
        """
        old_state = state
        state = tf.reshape(state, [-1, self._num_nodes, self._num_units])
        support = constants[0]

        output_size = 2 * self._num_units
        value = tf.sigmoid(
            self._gconv(inputs, state, support, output_size, bias_start=1.0)
        )
        value = tf.reshape(value, (-1, self._num_nodes, output_size))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)

        c = self._gconv(inputs, r * state, support, self._num_units)

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
    def _gconv(self, inputs, state, support, output_size, bias_start=0.0):

        batch_size = tf.shape(inputs)[0]
        inputs_and_state = tf.concat([inputs, state], axis=2)
        num_inpt_features = inputs_and_state.shape[-1]

        x = inputs_and_state

        x1 = tf.tensordot(x, support, axes=[1, 0])
        x = tf.reduce_sum(x1, axis=-1)

        x = tf.transpose(x1, [0, 2, 1])

        # x = tf.reshape(x, [batch_size, self._num_nodes, -1])

        if output_size == self._num_units:
            x = tf.matmul(x, self.w2) + self.b2
            # x = self.gconv_layer2(x)
        else:
            x = tf.matmul(x, self.w1) + self.b1
            # x = self.gconv_layer1(x)

        return x


class DCGRUBlock(tf_keras.layers.Layer):
    def __init__(self, dcrnn_cells, num_nodes, steps_to_predict, encode=True):
        super(DCGRUBlock, self).__init__()

        self.is_encoder = encode
        self.cells = dcrnn_cells
        self.num_nodes = num_nodes
        self.steps_to_predict = steps_to_predict
        # config.model.counter_position
        self.counter = tf.Variable(1, dtype=tf.float32, trainable=False)
        if encode:
            self.block = tf.keras.layers.RNN(self.cells, return_state=True)

    def encode(self, x, adj):
        state = self.block(x, constants=[adj])
        return state[1:]

    @tf.function
    def decay_teacher_coefficient(self):
        decay_rate = config.model.teacher_decay_rate

        teacher_coeff = decay_rate / (
            decay_rate + tf.exp(self.counter / decay_rate)
        )
        tf.summary.scalar(name="teacher_decay_coefficient", data=teacher_coeff)

        self.counter.assign_add(1)

        return teacher_coeff

    @tf.function
    def decode(self, state, adj, x_targ=None):

        batch_size = tf.shape(state[0])[0]

        init = tf.zeros([batch_size, self.num_nodes, 1], dtype=tf.float32)

        state = tuple(state)

        to_return = tf.TensorArray(
            size=self.steps_to_predict, dtype=tf.float32
        )
        if x_targ is None:
            for i in tf.range(self.steps_to_predict):
                init, state = self.cells(init, states=state, constants=[adj])
                to_return = to_return.write(i, init)
            return tf.transpose(to_return.stack(), [1, 0, 2, 3])
        else:
            for i in range(self.steps_to_predict):
                output, state = self.cells(init, states=state, constants=[adj])
                to_return = to_return.write(i, init)

                if (
                    tf.random.uniform(shape=[])
                    > self.decay_teacher_coefficient()
                ):
                    init = output
                else:
                    init = x_targ[:, i]

            return tf.transpose(to_return.stack(), [1, 0, 2, 3])

    def call(self, x, state, adj):
        if self.is_encoder:
            return self.encode(x, adj)
        else:
            return self.decode(state, adj, x)
