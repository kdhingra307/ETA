import tensorflow.keras as tf_keras
import numpy as np
from tensorflow.python.keras.backend import batch_normalization, dropout
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from ETA import config
from ETA.grud import GRUCell as GRUDCell

# from ETA.gru import GRUCell
from tensorflow import function as tf_function
import tensorflow as tf


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        # self.embedding = tf_keras.Sequential(
        #     [
        #         tf_keras.layers.Conv1D(
        #             filters=1024,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #         tf_keras.layers.BatchNormalization(),
        #         tf_keras.layers.Conv1D(
        #             filters=512,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #         tf_keras.layers.BatchNormalization(),
        #         tf_keras.layers.Conv1D(
        #             filters=256,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #         tf_keras.layers.BatchNormalization(),
        #         tf_keras.layers.Conv1D(
        #             filters=256,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #         tf_keras.layers.BatchNormalization(),
        #         tf_keras.layers.Conv1D(
        #             filters=512,
        #             kernel_size=3,
        #             padding="SAME",
        #             activation=tf_keras.layers.LeakyReLU(alpha=0.2),
        #         ),
        #     ]
        # )
        learnable_cell = GRUDCell(units=256)
        learnable_cell.build((None, None, 1166))

        self.encoder = tf_keras.layers.RNN(
            tf_keras.layers.StackedRNNCells(
                [
                    learnable_cell,
                    tf_keras.layers.GRUCell(
                        units=256,
                        # dropout=0.5,
                        # recurrent_dropout=0.5,
                        use_bias=False,
                    ),
                    # tf_keras.layers.GRUCell(units=128),
                ]
            ),
            return_state=True,
            name="encoding",
        )

        self.decoder = tf_keras.layers.StackedRNNCells(
            [
                tf_keras.layers.GRUCell(
                    units=256,
                    # dropout=0.5,
                    # recurrent_dropout=0.5,
                    use_bias=False,
                ),
                tf_keras.layers.GRUCell(
                    units=256,
                    # dropout=0.5,
                    # recurrent_dropout=0.5,
                    use_bias=False,
                ),
                # tf_keras.layers.GRUCell(units=128),
            ],
            name="decoding",
        )

        self.post_process = tf_keras.Sequential(
            [
                # tf_keras.layers.BatchNormalization(),
                # tf_keras.layers.Dropout(0.5),
                tf_keras.layers.Dense(
                    units=128,
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                # tf_keras.layers.BatchNormalization(),
                # tf_keras.layers.Dropout(0.5),
                tf_keras.layers.Dense(
                    units=256,
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.Dense(
                    units=config.model.num_nodes,
                    # use_bias=False,
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

            init = tf.zeros(
                [tf.shape(state[0])[0], num_nodes],
                dtype=tf.float32,
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

        return tf.transpose(to_return.stack(), [1, 0, 2])

    def call(self, x, training=False, y=None, constants=None):
        # embed = self.embedding(x)
        otpt = self.encoder(x, training=training, constants=constants)
        encoded = otpt[1:]
        decoded = self.decode(state=encoded, x_targ=y, training=training)
        return decoded

    def train_step(self, data):
        x, y, x2 = data
        print(x, y, x2)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1], constants=x2)
            # y_pred = self(x[:, :, :208], training=True, y=y[:, :, :, :1])
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        gradients = tape.gradient(loss, self.trainable_weights)
        gradients, norm = tf.clip_by_global_norm(gradients, 0.05)
        tf.print("norm---", norm, "---")

        my_zip = zip(self.trainable_weights, gradients)
        for weights, grads in my_zip:
            tf.summary.histogram(
                weights.name.replace(":", "_") + "_grads",
                data=grads,
            )

        my_zip = zip(gradients, self.trainable_weights)
        self.optimizer.apply_gradients(my_zip)

        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, x2 = data
        y_pred = self(x, training=False, constants=x2)
        # y_pred = self(x[:, :, :208], training=False)
        # Updates stateful loss metrics.
        loss = self.compiled_loss(
            y, y_pred, None, regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _, x2 = data
        return self(x, training=False, constants=x2)
