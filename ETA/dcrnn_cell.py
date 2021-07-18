import tensorflow.keras as tf_keras
import tensorflow as tf
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
    ):

        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []

        self.first_layer = GSConv(units=num_units * 2)
        self.second_layer = GSConv(num_units)

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
        else:
            self.ttr_counter = tf.Variable(
                1, dtype=tf.float32, trainable=False
            )
            self.ttr_val = tf.Variable(0.5, dtype=tf.float32, trainable=False)

    def build(self, x_shape):
        self.batch_size = x_shape[0]

    def encode(self, x, pos, z=None):
        state = self.block(
            x,
            constants=[pos, z],
            initial_state=(
                tf.zeros([tf.shape(x)[0], tf.shape(x)[2], 64]),
                tf.zeros([tf.shape(x)[0], tf.shape(x)[2], 64]),
            ),
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
    def decode(self, state, pos=None, x_targ=None, z=None, training=False):

        init = tf.zeros(
            [tf.shape(state[0])[0], tf.shape(state[0])[1], 1], dtype=tf.float32
        )

        state = tuple(state)

        to_return = tf.TensorArray(
            size=self.steps_to_predict, dtype=tf.float32
        )

        if x_targ is None:
            for i in tf.range(self.steps_to_predict):
                init, state = self.cells(
                    init, states=state, training=training, constants=[pos, z]
                )
                to_return = to_return.write(i, init)
        else:
            for i in tf.range(self.steps_to_predict):
                output, state = self.cells(
                    init, states=state, training=training, constants=[pos, z]
                )

                to_return = to_return.write(i, output)

                if tf.random.uniform(shape=[]) > self.ttr_val:
                    init = tf.stop_gradient(output)
                else:
                    init = x_targ[:, i]

        return tf.transpose(tf.squeeze(to_return.stack(), axis=-1), [1, 0, 2])

    def ttr(self):
        self.ttr_counter.assign_add(1)
        self.ttr_val.assign(tf.exp((-1 * self.ttr_counter) / (483 * 18)))

        tf.summary.scalar(
            "ttr_val",
            self.ttr_val,
            step=tf.cast(self.ttr_counter, tf.int64),
        )

    def call(self, x, state, pos, z):
        if self.is_encoder:
            return self.encode(x, pos=pos, z=z)
        else:
            return self.decode(state=state, x_targ=x, pos=pos, z=z)


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

        output = [x0]
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
