import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
import tensorflow.python.ops.array_ops as tf_array_ops
from tensorflow.python.keras.engine import data_adapter
import numpy as np
from ETA import config
import tensorflow.python.framework.dtypes as tf_dtype
from tensorflow import function as tf_function


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
                    activation=tf_keras.layers.LeakyReLU(alpha=0.2),
                ),
            ]
        )

    # @tf.function
    # def decay_teacher_coefficient(self):
    #     decay_rate = config.model.teacher_decay_rate

    #     teacher_coeff = decay_rate/ (decay_rate + self.counter/128)
    #     tf.summary.scalar(
    #         name="teacher_decay_coefficient",
    #         data=teacher_coeff,
    #         step=tf.cast(self.counter, tf.int64),
    #     )
    #     self.counter.assign_add(1)

    #     return teacher_coeff

    @tf_function
    def decode(self, state, x_targ=None, init=None):

        if init is None:
            num_nodes = config.model.num_nodes
            import tensorflow as tf

            init = tf_array_ops.zeros(
                [tf_array_ops.shape(state[0])[0], num_nodes],
                dtype=tf_dtype.float32,
            )

        tf.print(tf.shape(init))

        num_steps = config.model.steps_to_predict
        to_return = []
        if x_targ is None:
            for i in range(num_steps):
                init, state = self.decoder(init, states=state)
                init = self.post_process(init)
                to_return.append(init)
            return tf_array_ops.stack(to_return, axis=1)
        else:
            for i in range(num_steps):
                output, state = self.decoder(init, states=state)
                output = self.post_process(output)
                to_return.append(output)

                if np.random.rand() > config.model.ttr:
                    init = output
                else:
                    init = x_targ[:, i]

            return tf_array_ops.stack(to_return, axis=1)

    def call(self, x, training=False, y=None):
        embedding = self.embedding(x, training=training)
        otpt = self.encoder(embedding)
        encoded = otpt[1:]
        decoded = self.decode(state=encoded, x_targ=y)
        return decoded

    def train_step(self, data):
        x, y = data

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1])
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, None, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
