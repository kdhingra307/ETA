import tensorflow.keras as tf_keras
import numpy as np
from ETA import config
from tensorflow import function as tf_function
import tensorflow as tf
from ETA.gru import GRUCell


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.encoder = tf_keras.layers.RNN(
            tf_keras.layers.StackedRNNCells(
                [
                    GRUCell(units=128),
                    GRUCell(units=128),
                ]
            ),
            return_state=True,
            name="encoding",
        )

        self.decoder = tf_keras.layers.StackedRNNCells(
            [
                GRUCell(units=128),
                GRUCell(units=128),
            ],
            name="decoding",
        )

        self.post_process = tf_keras.Sequential(
            [
                tf_keras.layers.Dense(
                    units=64,
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.Dense(
                    units=32,
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
                tf_keras.layers.Dense(
                    units=1,
                ),
            ],
            name="post_process",
        )

    @tf_function
    def decode(self, state, x_targ=None, init=None, training=False):

        state = tuple(state)
        if init is None:
            import tensorflow as tf

            init = tf.zeros(
                [tf.shape(state[0])[0], 1],
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

        return tf.squeeze(tf.transpose(to_return.stack(), [1, 0, 2]), axis=-1)

    def call(self, x, training=False, y=None):

        otpt = self.encoder(x, training=training)
        encoded = otpt[1:]
        decoded = self.decode(state=encoded, x_targ=y, training=training)
        return decoded

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :1])
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        self.compiled_loss(y, y_pred, None, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
