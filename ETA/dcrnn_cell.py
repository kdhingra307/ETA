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
        etype=1,
        reuse=None,
        use_gc_for_ru=True,
    ):

        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru

        if etype == 1:
            self.first_layer = GSConv(units=num_units * 2, should=True)
            self.second_layer = GSConv(num_units, should=True)
        else:
            self.first_layer = GSConv(units=num_units * 2)
            self.second_layer = GSConv(num_units)

        if num_proj != None:
            self.projection_layer = tf_keras.Sequential(
                [
                    tf_keras.layers.Dense(
                        units=64,
                        activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                    ),
                    tf_keras.layers.BatchNormalization(),
                    tf_keras.layers.Dense(
                        units=32,
                        activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                    ),
                    tf_keras.layers.Dense(
                        units=1,
                    ),
                ]
            )

    # def build(self, inp_shape):
    #     inp_shape = list(inp_shape)
    #     inp_shape[-1] += self._num_units

    #     self.first_layer[0].build(inp_shape)
    #     inp_shape[-1] = self._num_units
    #     self.first_layer[1].build(inp_shape)

    #     self.second_layer[0].build(inp_shape)
    #     inp_shape[-1] = self._num_units
    #     self.second_layer[1].build(inp_shape)

    @tf.function
    def call(self, inputs, state, constants, training=False):

        """
            inputs_shape [BatchSize, Num_Nodes, Inp_features]
            state_shape [BatchSize, Num_Nodes, Num_units]

        Returns
        -------
        [type]
            [description]
        """
        support = constants[0]
        state = state[0]

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

        if self._num_proj is not None:
            output = self.projection_layer(output)

        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    # @tf.function
    # def _gconv(self, inputs, state, support, output_size, bias_start=0.0):

    #     print(output_size, self._num_units)
    #     if output_size == self._num_units:

    #     else:
    #         x = self.second_layer[0](inputs_and_state, support)
    #         x = self.second_layer[1](x, support)

    #     return x


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

    def encode(self, x, adj, training=False):
        state = self.block(x, training=training, constants=[adj])
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
    def decode(self, state, adj=None, x_targ=None, training=False):

        batch_size = tf.shape(state[0])[0]

        init = tf.zeros([batch_size, self.num_nodes, 1], dtype=tf.float32)

        state = tuple(state)

        to_return = tf.TensorArray(
            size=self.steps_to_predict, dtype=tf.float32
        )
        for i in tf.range(self.steps_to_predict):
            init, state = self.cells(
                init, states=state, training=training, constants=[adj]
            )
            to_return = to_return.write(i, init)
        return tf.transpose(to_return.stack(), [1, 0, 2, 3])

    def call(self, x, state, adj=None, training=False):
        if self.is_encoder:
            return self.encode(x, adj, training=training)
        else:
            return self.decode(state, adj, x, training=training)


class GSConv(tf_keras.layers.Layer):
    def __init__(self, units, should=False):
        super(GSConv, self).__init__()

        self.layer = tf.keras.layers.Dense(
            units=units,
            activation=tf.keras.layers.LeakyReLU(0.2),
        )
        self.layer1 = tf.keras.layers.Dense(
            units=units,
            activation=tf.keras.layers.LeakyReLU(0.2),
        )
        self.layer2 = tf.keras.layers.Dense(units=units)
        self.should = should

    def call(self, x0, support, training=False):

        # size = 1
        # if self.should:
        #     x = tf.tensordot(support[1], x0, axes=[1, 1])
        #     x = tf.transpose(x, [1, 0, 2])

        #     return self.layer(x, training=training)
        # else:

        output = []
        x = tf.tensordot(support[:1], x0, axes=[1, 1])
        x = tf.transpose(x, [2, 1, 3, 0])
        x = tf.reshape(x, [tf.shape(x0)[0], tf.shape(x0)[1], x0.shape[-1]])
        x = self.layer(x)
        output.append(x)

        x = tf.tensordot(
            support[1:2], tf.concat([x, x0], axis=-1), axes=[1, 1]
        )
        x = tf.transpose(x, [2, 1, 3, 0])
        x = tf.reshape(
            x, [tf.shape(x0)[0], tf.shape(x0)[1], x.shape[-1] + x0.shape[-1]]
        )

        x = self.layer1(x)
        output.append(x)

        return self.layer2(tf.concat(output, axis=-1))
