import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
import tensorflow.python.ops.array_ops as tf_array_ops
from tensorflow.python.keras.engine import data_adapter
import numpy as np
from ETA import config
import tensorflow.python.framework.dtypes as tf_dtype
from tensorflow import function as tf_function
import tensorflow as tf
from ETA import GRUDCell


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.time_missing = tf_keras.layers.GRU(128, return_sequences=True)

        # self.embedding = tf_keras.Sequential(
        #     [
        #         tf_keras.layers.Conv1D(
        #             filters=128,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #         tf_keras.layers.BatchNormalization(),
        #         tf_keras.layers.Conv1D(
        #             filters=64,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #         tf_keras.layers.BatchNormalization(),
        #         tf_keras.layers.Conv1D(
        #             filters=128,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #         tf_keras.layers.BatchNormalization(),
        #         tf_keras.layers.Conv1D(
        #             filters=64,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #         tf_keras.layers.BatchNormalization(),
        #         tf_keras.layers.Conv1D(
        #             filters=128,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #     ],
        #     name="embedding",
        # )
        # learnable_cell = GRUDCell(units=64)
        # learnable_cell.build((None, None, 208))
        self.encoder = tf_keras.layers.RNN(
            tf_keras.layers.StackedRNNCells(
                [
                    tf_keras.layers.GRUCell(units=64),
                    tf_keras.layers.GRUCell(
                        units=64, dropout=0.25, recurrent_dropout=0.25
                    ),
                ]
            ),
            return_state=True,
            name="encoding",
        )

        self.decoder = tf_keras.layers.StackedRNNCells(
            [
                tf_keras.layers.GRUCell(units=64),
                tf_keras.layers.GRUCell(
                    units=64, dropout=0.25, recurrent_dropout=0.25
                ),
            ],
            name="decoding",
        )

        self.post_process = tf_keras.Sequential(
            [
                # tf_keras.layers.Dense(
                #     units=128,
                #     activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                # ),
                # tf_keras.layers.BatchNormalization(),
                # tf_keras.layers.Dropout(0.5),
                # tf_keras.layers.Dense(
                #     units=256,
                #     activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                # ),
                tf_keras.layers.Dense(
                    units=207,
                ),
            ],
            name="post_process",
        )

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

                init = tf.stop_gradient(output)

        return tf_array_ops.transpose(to_return.stack(), [1, 0, 2])

    def call(self, x, training=False, y=None, constants=None):

        # pre_embedding = self.time_missing(x)
        # print(pre_embedding.shape)
        # embedding = self.embedding(x, training=training)
        otpt = self.encoder(x, training=training, constants=constants)
        encoded = otpt[1:]
        decoded = self.decode(state=encoded, x_targ=y, training=training)
        return decoded

    def train_step(self, data):
        x, y, x2 = data

        with tf_diff.GradientTape() as tape:
            # y_pred = self(x, training=True, y=y[:, :, :, :1], constants=x2)
            y_pred = self(x[:, :, :208], training=True, y=y[:, :, :, :1])
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, x2 = data
        # y_pred = self(x, training=False, constants=x2)
        y_pred = self(x[:, :, :208], training=False)
        # Updates stateful loss metrics.
        loss = self.compiled_loss(
            y, y_pred, None, regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _, x2 = data
        return self(x, training=False, constants=x2)
