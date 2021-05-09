import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
import tensorflow.python.ops.array_ops as tf_array_ops
from tensorflow.python.keras.engine import data_adapter
import numpy as np
from ETA import config
import tensorflow.python.framework.dtypes as tf_dtype
from tensorflow import function as tf_function
import tensorflow as tf


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.embedding = tf_keras.Sequential(
            [
                tf_keras.layers.Conv1D(
                    filters=128,
                    kernel_size=3,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Conv1D(
                    filters=64,
                    kernel_size=3,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Conv1D(
                    filters=128,
                    kernel_size=3,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Conv1D(
                    filters=64,
                    kernel_size=3,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Conv1D(
                    filters=128,
                    kernel_size=3,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
            ]
        )

        self.encoder = tf_keras.layers.RNN(
            tf_keras.layers.StackedRNNCells(
                [
                    tf_keras.layers.GRUCell(units=256),
                    tf_keras.layers.GRUCell(units=128),
                ]
            ),
            return_state=True,
        )

        self.decoder = tf_keras.layers.StackedRNNCells(
            [
                tf_keras.layers.GRUCell(units=256),
                tf_keras.layers.GRUCell(units=128),
            ]
        )

        self.post_process = tf_keras.Sequential(
            [
                tf_keras.layers.Dense(
                    units=128,
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Dense(
                    units=256,
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Dense(
                    units=207,
                ),
            ]
        )

        self.q_table = tf.Variable(
            tf.zeros([100, 3]), dtype=tf.float32, trainable=False
        )
        self.ttr_param = config.model.ttr
        self.prev_q_state = tf.Variable(
            config.model.ttr, dtype=tf.int32, trainable=False
        )
        tf.print(self.prev_q_state)
        self.counter = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.avg_train = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.gcounter = tf.Variable(0, dtype=tf.int64, trainable=False)

    @tf.function
    def q_update_train(self, loss):

        self.counter.assign_add(1)
        self.avg_train.assign_add(loss)

    def q_update_val(self, loss):

        action = tf.cond(
            tf.random.uniform(shape=[]) < 0.01,
            lambda: tf.random.uniform(
                shape=[], minval=0, maxval=3, dtype=tf.int32
            ),
            lambda: tf.argmax(
                self.q_table[self.prev_q_state], output_type=tf.int32
            ),
        )

        next_state = self.prev_q_state.value() + tf.cond(
            (action == 0),
            lambda: -1,
            lambda: tf.cond(
                action == 1,
                lambda: 0,
                lambda: 1,
            ),
        )

        next_state = tf.cond(next_state < 0, lambda: 1, lambda: next_state)
        next_state = tf.cond(next_state >= 100, lambda: 98, lambda: next_state)

        average_train = self.avg_train / self.counter

        # tf.print(
        #     action,
        #     action == 1,
        #     action == 2,
        #     action == 0,
        #     self.prev_q_state,
        #     next_state,
        # )

        tf.print(next_state, self.prev_q_state, action)

        self.q_table.scatter_nd_add(
            tf.expand_dims(
                tf.stack([self.prev_q_state, action], axis=-1), axis=0
            ),
            tf.expand_dims(
                0.8
                * (
                    (average_train - loss)
                    + 0.95 * tf.reduce_max(self.q_table[next_state])
                    - self.q_table[self.prev_q_state, action]
                ),
                axis=0,
            ),
        )

        self.prev_q_state.assign(next_state)

        self.ttr_param = next_state / 100

        tf.summary.scalar(
            name="Q/ttr",
            data=self.ttr_param,
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/table",
            data=tf.reduce_mean(self.q_table),
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/reward",
            data=(average_train - loss),
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/train",
            data=(average_train),
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/loss",
            data=(loss),
            step=self.gcounter,
        )
        self.counter.assign_add(1)
        self.gcounter.assign_add(1)

    @tf_function
    def decode(self, state, x_targ=None, init=None, training=False):

        if init is None:
            num_nodes = config.model.num_nodes
            import tensorflow as tf

            init = tf_array_ops.zeros(
                [tf_array_ops.shape(state[0])[0], num_nodes],
                dtype=tf_dtype.float32,
            )

        num_steps = config.model.steps_to_predict
        to_return = []
        if x_targ is None:
            for i in range(num_steps):
                init, state = self.decoder(
                    init, states=state, training=training
                )
                init = self.post_process(init, training=training)
                to_return.append(init)
            return tf_array_ops.stack(to_return, axis=1)
        else:
            for i in range(num_steps):
                output, state = self.decoder(
                    init, states=state, training=training
                )
                output = self.post_process(output, training=training)
                to_return.append(output)

                if tf.random.uniform(shape=[]) < self.ttr_param:
                    init = output
                else:
                    init = tf_array_ops.squeeze(x_targ[:, i], axis=-1)

            return tf_array_ops.stack(to_return, axis=1)

    def call(self, x, training=False, y=None):

        embedding = self.embedding(x, training=training)
        otpt = self.encoder(embedding, training=training)
        encoded = otpt[1:]
        decoded = self.decode(state=encoded, x_targ=y, training=training)
        return decoded

    def train_step(self, data):
        x, y = data

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1])
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        self.q_update_train(loss)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        loss = self.compiled_loss(
            y, y_pred, None, regularization_losses=self.losses
        )
        self.q_update_val(loss)
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
