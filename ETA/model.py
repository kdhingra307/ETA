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

        self.layer = [
            tf.keras.Sequential(
                [
                    # tf.keras.layers.Dense(
                    #     units=2 * units, activation=tf.keras.layers.LeakyReLU()
                    # ),
                    # tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(
                        filters=units,
                        activation=tf.keras.layers.LeakyReLU(0.2),
                        kernel_size=[3, 1],
                        padding="Same",
                        strides=[1, 1],
                    ),
                ]
            )
            for _ in range(4)
        ]

    def operation(self, x0, support, layer, training=False):
        x = tf.tensordot(support, x0, axes=[1, 2])
        x = tf.transpose(x, [1, 2, 0, 3])

        return layer(x, training=training)

    def call(self, x, support, training=False):
        output = []
        x = self.operation(x, support[0], self.layer[0], training=training)
        for i in range(1, 4):
            x += self.operation(
                x, support[i], self.layer[i], training=training
            )
            # output.append(x)

        return x


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

        # x = self.gconv(x, adj, training=training)

        encoded = self.encoder(x=x, adj=adj, state=None, training=training)
        decoded = self.decoder(adj=adj, state=encoded, x=y, training=training)
        return tf_squeeze(decoded, axis=-1)

    def train_step(self, data):
        pos, x, y, z = data

        sample_weight = None

        with tf_diff.GradientTape() as tape:
            y_pred = self(
                x[:, :, :, :2], training=True, y=y[:, :, :, :1], adj=pos, z=z
            )
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        pos, x, y, z = data
        y_pred = self(x[:, :, :, :2], training=False, adj=pos, z=z)
        # Updates stateful loss metrics.
        loss = self.compiled_loss(
            y, y_pred, None, regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
