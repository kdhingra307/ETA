import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
import tensorflow.python.ops.array_ops as tf_array_ops
from tensorflow.python.keras.engine import data_adapter
import numpy as np
from ETA import config, loss_function
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
                    tf_keras.layers.GRUCell(units=64),
                    tf_keras.layers.GRUCell(units=64),
                ]
            ),
            return_state=True,
        )

        self.decoder = tf_keras.layers.StackedRNNCells(
            [
                tf_keras.layers.GRUCell(units=64),
                tf_keras.layers.GRUCell(units=64),
            ]
        )

        self.post_process = self.post_process = tf_keras.Sequential(
            [
                # tf_keras.layers.Dense(
                #     units=128,
                #     activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                # ),
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
        self.dcounter = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.error = {
            "mse": loss_function,
            "gan": tf_keras.losses.BinaryCrossentropy(from_logits=True),
        }
        self.adversarial = tf.Variable(False, dtype=tf.bool, trainable=False)

    def build(self, input_shape):
        super().build(input_shape)

        self.generator_variables = (
            self.embedding.trainable_variables
            + self.encoder.trainable_variables
            + self.decoder.trainable_variables
            + self.post_process.trainable_variables
        )

    @tf_function
    def decode(self, state, x_targ=None, init=None, training=False):

        state = tuple(state)

        num_nodes = config.model.num_nodes
        init = tf_array_ops.zeros(
            [tf_array_ops.shape(state[0])[0], num_nodes],
            dtype=tf_dtype.float32,
        )

        num_steps = config.model.steps_to_predict
        to_return = tf.TensorArray(size=num_steps, dtype=tf.float32)

        for i in tf.range(num_steps):
            init, state = self.decoder(init, states=state, training=training)

            init = self.post_process(init, training=training)
            to_return = to_return.write(i, init)

        return (tf_array_ops.transpose(to_return.stack(), [1, 0, 2]),)

    def call(self, x, training=False, y=None):

        embedding = self.embedding(x, training=training)
        otpt = self.encoder(embedding, training=training)
        encoded = otpt[1:]
        decoded = self.decode(state=encoded, x_targ=y, training=training)
        return decoded

    def auto_regression(self, x, y_label, training=False):

        y_out = self(x, training=training)

        gen_loss = self.compiled_loss(
            y_true=y_label,
            y_pred=y_out,
            sample_weight=None,
            regularization_losses=self.losses,
        )

        return (
            gen_loss,
            y_label,
            y_out,
        )

    def teacher_force(self, x, y_true, training=False):

        y_out, embedding = self(x, training=training, y=y_true[:, :, :, :1])

        discriminator = tf.squeeze(self.discriminator(embedding), axis=-1)

        mse_loss = self.error["mse"](y_true=y_true, y_pred=y_out)
        gen_loss = self.error["gan"](
            y_true=tf.ones(tf.shape(discriminator)[0]), y_pred=discriminator
        )
        dis_loss = self.error["gan"](
            y_true=tf.zeros(tf.shape(discriminator)[0]), y_pred=discriminator
        )
        return (
            tf.cond(
                self.adversarial, lambda: gen_loss + mse_loss, lambda: mse_loss
            ),
            dis_loss,
            {
                "seq2seq/ttf": y_true,
                "gan/ttf": tf.zeros(tf.shape(discriminator)[0]),
            },
            {
                "seq2seq/ttf": y_out,
                "gan/ttf": discriminator,
            },
        )

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf_diff.GradientTape() as tape1:

            gloss2, mtrue2, mpred2 = self.auto_regression(x, y, training=True)

            gloss = gloss2

        self.optimizer.minimize(gloss, self.generator_variables, tape=tape1)

        self.compiled_metrics.update_state(
            mtrue2,
            mpred2,
            None,
        )

        self.dcounter.assign_add(1)

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
