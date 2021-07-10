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
        print("ipp", inputs)
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


class DCGRUBlock(tf_keras.layers.Layer):
    def __init__(self, dcrnn_cells, num_nodes, steps_to_predict, encode=True):
        super(DCGRUBlock, self).__init__()

        self.is_encoder = encode
        self.cells = dcrnn_cells
        self.num_nodes = num_nodes
        self.steps_to_predict = steps_to_predict
        # config.model.counter_position
        if encode:
            self.block = tf.keras.layers.RNN(self.cells, return_state=True)
        else:
            self.ttr_counter = tf.Variable(
                0, dtype=tf.float32, trainable=False
            )
            self.ttr_val = tf.Variable(0, dtype=tf.float32, trainable=False)

    def encode(self, x, adj, training=False, z=None):
        print(x, adj, z)
        state = self.block(x, training=training, constants=[adj, z])
        return state[1:]

    @tf.function
    def decode(self, state, adj=None, x_targ=None, training=False, z=None):
        batch_size = tf.shape(state[0])[0]

        init = tf.zeros([batch_size, self.num_nodes, 1], dtype=tf.float32)

        state = tuple(state)

        to_return = tf.TensorArray(
            size=self.steps_to_predict, dtype=tf.float32
        )

        if x_targ is None:
            for i in tf.range(self.steps_to_predict):
                init, state = self.cells(
                    init, states=state, training=training, constants=[adj, z]
                )
                to_return = to_return.write(i, init)
        else:
            for i in tf.range(self.steps_to_predict):
                output, state = self.cells(
                    init, states=state, training=training, constants=[adj, z]
                )

                to_return = to_return.write(i, output)

                if tf.random.uniform(shape=[]) < self.ttr_val:
                    init = tf.stop_gradient(output)
                else:
                    init = x_targ[:, i]

    def ttr(self):
        self.ttr_counter.assign_add(1)
        if self.ttr_counter > 483 * 3:
            self.ttr_val.assign(tf.sin((50 * 483) / self.ttr_counter))
        tf.summary.scalar("ttr_val", self.ttr_val)

    def call(self, x, state, adj=None, training=False, z=None):
        if self.is_encoder:
            return self.encode(
                x,
                adj,
                training=training,
                z=z,
            )
        else:
            return self.decode(state, adj, training=training, z=z, x_targ=x)


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
