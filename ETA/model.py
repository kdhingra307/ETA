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

        self.discriminator = tf_keras.layers.RNN(
            tf_keras.layers.StackedRNNCells(
                [
                    tf_keras.layers.GRUCell(units=32),
                    tf_keras.layers.GRUCell(units=1),
                ]
            )
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
        self.dcounter = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.error = {
            "mse": loss_function,
            "gan": tf_keras.losses.BinaryCrossentropy(from_logits=True),
        }

    def build(self, input_shape):
        super().build(input_shape)

        self.generator_variables = (
            self.embedding.trainable_variables
            + self.encoder.trainable_variables
            + self.decoder.trainable_variables
            + self.post_process.trainable_variables
        )

        self.discriminator.build(
            tf.TensorShape([input_shape[0], input_shape[1], 128])
        )

        self.discriminator_variables = self.discriminator.trainable_variables

    @tf_function
    def decode(self, state, x_targ=None, init=None, training=False):

        state = tuple(state)

        if init is None:
            num_nodes = config.model.num_nodes
            init = tf_array_ops.zeros(
                [tf_array_ops.shape(state[0])[0], num_nodes],
                dtype=tf_dtype.float32,
            )

        num_steps = config.model.steps_to_predict
        to_return = tf.TensorArray(size=num_steps, dtype=tf.float32)
        embedding = tf.TensorArray(size=num_steps, dtype=tf.float32)

        if x_targ is None:
            for i in tf.range(num_steps):
                init, state = self.decoder(
                    init, states=state, training=training
                )

                embedding = embedding.write(i, init)
                init = self.post_process(init, training=training)
                to_return = to_return.write(i, init)

        else:
            for i in tf.range(num_steps):
                output, state = self.decoder(
                    init, states=state, training=training
                )

                embedding = embedding.write(i, output)
                output = self.post_process(output, training=training)
                to_return = to_return.write(i, output)

                init = tf_array_ops.squeeze(x_targ[:, i], axis=-1)

        return (
            tf_array_ops.transpose(to_return.stack(), [1, 0, 2]),
            tf_array_ops.transpose(embedding.stack(), [1, 0, 2]),
        )

    def call(self, x, training=False, y=None):

        embedding = self.embedding(x, training=training)
        otpt = self.encoder(embedding, training=training)
        encoded = otpt[1:]
        decoded = self.decode(state=encoded, x_targ=y, training=training)
        return decoded

    def auto_regression(self, x, y_true, training=False):

        y_out, embedding = self(x, training=training)

        discriminator = tf.squeeze(
            self.discriminator(tf.stop_gradient(embedding)), axis=-1
        )
        generator = tf.squeeze(self.discriminator(embedding), axis=-1)

        gen_loss = self.error["mse"](y_true=y_true, y_pred=y_out) + self.error[
            "gan"
        ](
            y_true=tf.zeros(tf.shape(discriminator)[0]),
            y_pred=generator,
        )

        dis_loss = self.error["gan"](
            y_true=tf.ones(tf.shape(discriminator)[0]), y_pred=discriminator
        )

        return (
            gen_loss,
            dis_loss,
            {
                "seq2seq": y_true,
                "generator": tf.zeros(tf.shape(discriminator)[0]),
                "discriminator": tf.ones(tf.shape(discriminator)[0]),
            },
            {
                "seq2seq": y_out,
                "generator": generator,
                "discriminator": discriminator,
            },
        )

    def teacher_force(self, x, y_true, training=False):

        y_out, embedding = self(x, training=training, y=y_true[:, :, :, :1])

        discriminator = tf.squeeze(
            self.discriminator(tf.stop_gradient(embedding)), axis=-1
        )

        gen_loss = self.error["mse"](y_true=y_true, y_pred=y_out)
        dis_loss = self.error["gan"](
            y_true=tf.zeros(tf.shape(discriminator)[0]), y_pred=discriminator
        )
        return (
            gen_loss,
            dis_loss,
            {
                "seq2seq": y_true,
                "generator": tf.zeros(tf.shape(discriminator)[0]),
                "discriminator": tf.zeros(tf.shape(discriminator)[0]),
            },
            {
                "seq2seq": y_out,
                "generator": tf.zeros(tf.shape(discriminator)[0]),
                "discriminator": discriminator,
            },
        )

    def train_step(self, data):
        x, y = data

        with tf_diff.GradientTape() as tape1, tf_diff.GradientTape() as tape2:

            gloss, dloss, mtrue, mpred = tf.cond(
                self.dcounter % 2 == 0,
                lambda: self.teacher_force(x, y, training=True),
                lambda: self.auto_regression(x, y, training=True),
            )

        self.optimizer["generator"].minimize(
            gloss, self.generator_variables, tape=tape1
        )
        print(self.discriminator_variables)
        self.optimizer["discriminator"].minimize(
            dloss, self.discriminator_variables, tape=tape2
        )

        tf.cond(
            self.dcounter % 2 == 0,
            lambda: self.compiled_metrics.update_state(
                {
                    "seq2seq/ttf": mtrue["seq2seq"],
                    "discriminator/ttf": mtrue["discriminator"],
                    "seq2seq/ar": None,
                    "discriminator/ar": None,
                    "generator/ar": None,
                },
                {
                    "seq2seq/ttf": mpred["seq2seq"],
                    "discriminator/ttf": tf.nn.sigmoid(mpred["discriminator"]),
                    "seq2seq/ar": None,
                    "discriminator/ar": None,
                    "generator/ar": None,
                },
                None,
            ),
            lambda: self.compiled_metrics.update_state(
                {
                    "seq2seq/ar": mtrue["seq2seq"],
                    "discriminator/ar": mtrue["discriminator"],
                    "generator/ar": mtrue["generator"],
                    "seq2seq/ttf": None,
                    "discriminator/ttf": None,
                },
                {
                    "seq2seq/ar": mpred["seq2seq"],
                    "discriminator/ar": tf.nn.sigmoid(mpred["discriminator"]),
                    "generator/ar": tf.nn.sigmoid(mpred["generator"]),
                    "seq2seq/ttf": None,
                    "discriminator/ttf": None,
                },
                None,
            ),
        )

        self.dcounter.assign_add(1)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred, _ = self(x, training=False)
        # Updates stateful loss metrics.
        # loss = self.compiled_loss(
        #     {"mse": y, "generator": None, "discriminator": None},
        #     {"mse": y_pred, "generator": None, "discriminator": None},
        #     None,
        #     regularization_losses=self.losses,
        # )

        # # self.q_update_val(loss)
        self.compiled_metrics.update_state(
            {
                "seq2seq/ar": y,
                "discriminator/ar": None,
                "generator/ar": None,
                "seq2seq/ttf": None,
                "discriminator/ttf": None,
            },
            {
                "seq2seq/ar": y_pred,
                "discriminator/ar": None,
                "generator/ar": None,
                "seq2seq/ttf": None,
                "discriminator/ttf": None,
            },
            None,
        )
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
