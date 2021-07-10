import tensorflow.keras as tf_keras
from tensorflow.python.keras.engine.sequential import Sequential
import tensorflow.autodiff as tf_diff
from tensorflow import squeeze as tf_squeeze
from tensorflow.python.keras.engine import data_adapter
from ETA import DCGRUBlock, DCGRUCell, config
import numpy as np
import tensorflow as tf


class GConv(tf_keras.layers.Layer):
    def __init__(self, units):
        super(GConv, self).__init__()

        self.x_prev1 = [
            tf.keras.layers.Conv2D(
                filters=8,
                activation=tf.keras.layers.LeakyReLU(0.2),
                kernel_size=[4, 1],
                padding="Same",
                strides=[1, 1],
            )
            for _ in range(3)
        ]

        self.x_prev = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=[4, 1],
            padding="Same",
            strides=[1, 1],
        )

        self.h_prev = tf.keras.layers.Dense(units, name="h_prev")

        # self.layer = [
        #     tf.keras.Sequential(
        #         [
        #             tf.keras.layers.Conv2D(
        #                 filters=units,
        #                 activation=tf.keras.layers.LeakyReLU(0.2),
        #                 kernel_size=[3, 1],
        #                 padding="Same",
        #                 strides=[1, 1],
        #             ),
        #         ]
        #     )
        #     for _ in range(2)
        # ]

    def operation(self, x0, support, layer, training=False):
        x = tf.tensordot(support, x0, axes=[1, 2])
        x = tf.transpose(x, [1, 2, 0, 3])

        return layer(x, training=training)

    def call(self, x, support, x2, training=False):

        x, x1, mask, dt = tf.split(x, num_or_size_splits=4, axis=-1)

        embedding = self.x_prev1[0](dt)
        embedding += self.x_prev1[1](embedding)
        embedding += self.x_prev1[2](embedding)
        embedding = tf.concate([embedding, dt], axis=-1)

        x_prev_mask = tf.exp(
            -1 * tf.clip_by_value(self.x_prev(embedding), 0, tf.float32.max)
        )
        x2 = tf.expand_dims(x2, axis=1)

        x = (x * mask) + (
            (1 - mask) * (x_prev_mask * x1 + (1 - x_prev_mask) * x2)
        )

        return x

        # h_prev_mask = tf.exp(
        #     -1 * tf.clip_by_value(self.h_prev(dt), 0, tf.float32.max)
        # )

        # output = []
        # for i in range(0, 2):
        #     cur_otpt = self.operation(
        #         x, support[i], self.layer[i], training=training
        #     )
        #     output.append(cur_otpt)

        # return tf.concat(output, axis=-1)


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()
        num_nodes = config.model.graph_batch_size
        steps_to_predict = config.model.steps_to_predict

        self.encoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [
                    DCGRUCell(128, 2, num_nodes, etype=1),
                    DCGRUCell(128, 2, num_nodes, etype=1),
                ]
            ),
            num_nodes=num_nodes,
            steps_to_predict=steps_to_predict,
        )

        self.decoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [
                    DCGRUCell(128, 2, num_nodes, etype=0),
                    DCGRUCell(128, 2, num_nodes, num_proj=1, etype=0),
                ]
            ),
            num_nodes=num_nodes,
            steps_to_predict=steps_to_predict,
            encode=False,
        )

        self.gconv = GConv(32)

    def call(self, x, training=False, y=None, adj=None, z=None):

        x = self.gconv(x, adj, x2=z, training=training)

        encoded = self.encoder(x=x, adj=adj, state=None, training=training)
        decoded = self.decoder(adj=adj, state=encoded, x=y, training=training)
        return tf_squeeze(decoded, axis=-1)

    def train_step(self, data):
        pos, x, y, z = data

        sample_weight = None

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1], adj=pos, z=z)
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        pos, x, y, z = data
        y_pred = self(x, training=False, adj=pos, z=z)
        # Updates stateful loss metrics.
        loss = self.compiled_loss(
            y, y_pred, None, regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
