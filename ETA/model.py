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
                    filters=512,
                    kernel_size=3,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Conv1D(
                    filters=256,
                    kernel_size=2,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Conv1D(
                    filters=256,
                    kernel_size=2,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Conv1D(
                    filters=512,
                    kernel_size=3,
                    padding="SAME",
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
            ],
            name="embedding",
        )

        self.encoder = tf_keras.layers.RNN(
            tf_keras.layers.StackedRNNCells(
                [
                    tf_keras.layers.GRUCell(units=256),
                    tf_keras.layers.GRUCell(
                        units=512, dropout=0.1, recurrent_dropout=0.1
                    ),
                ]
            ),
            return_state=True,
            name="encoding",
        )

        self.decoder = tf_keras.layers.StackedRNNCells(
            [
                tf_keras.layers.GRUCell(units=256),
                tf_keras.layers.GRUCell(
                    units=512, dropout=0.1, recurrent_dropout=0.1
                ),
            ],
            name="decoding",
        )

        self.post_process = tf_keras.Sequential(
            [
                tf_keras.layers.Dense(
                    units=1024,
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Dropout(0.5),
                tf_keras.layers.Dense(
                    units=2048,
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.Dense(
                    units=config.model.num_nodes,
                ),
            ],
            name="post_process",
        )

        self.q_table = tf.Variable(
            tf.zeros([101, 3]), dtype=tf.float32, trainable=False
        )
        self.ttr_param2 = tf.Variable(
            tf.cast(config.model.ttr, tf.float32),
            dtype=tf.float32,
            trainable=False,
        )
        self.ttr_param = tf.Variable(
            tf.cast(config.model.ttr, tf.float32),
            dtype=tf.float32,
            trainable=False,
        )
        self.prev_q_state = tf.Variable(
            tf.cast(100 * config.model.ttr, tf.int32),
            dtype=tf.int32,
            trainable=False,
        )

        self.counter = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.avg_train = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.gcounter = tf.Variable(1, dtype=tf.int64, trainable=False)
        self.current_reward = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.dcounter = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.accuracy_metric = tf.keras.metrics.Accuracy()
        self.disc_loss = tf_keras.losses.BinaryCrossentropy(from_logits=True)
        self.ttf_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.ar_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    @tf.function
    def q_update_train(self, loss):

        self.ttf_loss.assign(
            tf.cond(
                self.gcounter % 3 == 1, lambda: loss, lambda: self.ttf_loss
            )
        )
        self.ar_loss.assign(
            tf.cond(self.gcounter % 3 == 0, lambda: loss, lambda: self.ar_loss)
        )

        tf.cond(
            self.gcounter % 3 == 2,
            lambda: self.q_update_val(loss),
            lambda: None,
        )

        self.gcounter.assign_add(1)

    def q_update_val(self, loss):

        epsilon = 0.9 / (1 + tf.cast(10 * self.gcounter, tf.float32) / 10000)

        action = tf.cond(
            tf.random.uniform(shape=[]) < epsilon,
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

        self.ttr_param2.assign(tf.cast(next_state / 100, tf.float32))

        relative_ttf = tf.abs(loss - self.ttf_loss)
        relative_ar = tf.abs(loss - self.ar_loss)

        relative = tf.cond(
            relative_ttf > relative_ar,
            lambda: tf.clip_by_value(relative_ttf / relative_ar, 1, 2),
            lambda: tf.clip_by_value(relative_ar / relative_ttf, 1, 2),
        )

        self.q_table.scatter_nd_add(
            tf.expand_dims(
                tf.stack([self.prev_q_state, action], axis=-1), axis=0
            ),
            tf.expand_dims(
                0.6
                * (
                    -1 * (relative - 1.25)
                    + 0.9 * tf.reduce_max(self.q_table[next_state])
                    - self.q_table[self.prev_q_state, action]
                ),
                axis=0,
            ),
        )

        self.prev_q_state.assign(next_state)

        tf.summary.scalar(
            name="Q/ttr",
            data=self.ttr_param2,
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/assert",
            data=epsilon,
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/table",
            data=tf.reduce_mean(self.q_table),
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/reward",
            data=relative,
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/ttf",
            data=(relative_ttf),
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/Ar",
            data=(relative_ar),
            step=self.gcounter,
        )
        tf.summary.scalar(
            name="Q/loss",
            data=(loss),
            step=self.gcounter,
        )
        self.gcounter.assign_add(1)

    @tf_function
    def decode(self, state, x_targ=None, init=None, training=False):

        state = tuple(state)
        if init is None:
            num_nodes = config.model.num_nodes
            import tensorflow as tf

            init = tf_array_ops.zeros(
                [tf_array_ops.shape(state[0])[0], num_nodes],
                dtype=tf_dtype.float32,
            )

        num_steps = config.model.steps_to_predict
        to_return = tf.TensorArray(size=num_steps, dtype=tf.float32)
        if x_targ is None:
            for i in tf.range(num_steps):
                init, state = self.decoder(
                    init, states=state, training=training
                )
                init = self.post_process(init, training=training)
                to_return = to_return.write(i, init)
        else:
            for i in tf.range(num_steps):
                output, state = self.decoder(
                    init, states=state, training=training
                )
                output = self.post_process(output, training=training)
                to_return = to_return.write(i, output)

                if tf.random.uniform(shape=[]) < self.ttr_param:
                    init = tf.stop_gradient(output)
                else:
                    init = tf_array_ops.squeeze(x_targ[:, i], axis=-1)

        return tf_array_ops.transpose(to_return.stack(), [1, 0, 2])

    def call(self, x, training=False, y=None):

        embedding = self.embedding(x, training=training)
        otpt = self.encoder(embedding, training=training)
        encoded = otpt[1:]
        decoded = self.decode(state=encoded, x_targ=y, training=training)
        return decoded

    def train_step(self, data):
        x, y = data

        # self.ttr_param.assign(
        #     tf.cond(
        #         self.gcounter % 3 == 0,
        #         lambda: 1.0,
        #         lambda: tf.cond(
        #             self.gcounter % 3 == 1,
        #             lambda: 0.0,
        #             lambda: self.ttr_param2,
        #         ),
        #     )
        # )

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1])
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        # self.q_update_train(loss)

        # self.ttr_param.assign(
        #     tf.cast(config.model.ttr, tf.float32)
        #     / (1 + tf.cast(10 * self.gcounter, tf.float32) / 50000)
        # )

        tf.summary.scalar(
            name="Q/ttr",
            data=self.ttr_param,
            step=self.gcounter,
        )
        self.gcounter.assign_add(1)

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
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
