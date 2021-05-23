import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
from tensorflow import squeeze as tf_squeeze
from tensorflow.python.keras.engine import data_adapter
from ETA import DCGRUBlock, DCGRUCell, config
import numpy as np
import tensorflow as tf


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()
        num_nodes = config.model.graph_batch_size
        steps_to_predict = config.model.steps_to_predict

        self.encoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [DCGRUCell(256, 2, num_nodes), DCGRUCell(128, 2, num_nodes)]
            ),
            num_nodes=num_nodes,
            steps_to_predict=steps_to_predict,
        )

        self.decoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [
                    DCGRUCell(256, 2, num_nodes),
                    DCGRUCell(128, 2, num_nodes, num_proj=1),
                ]
            ),
            num_nodes=num_nodes,
            steps_to_predict=steps_to_predict,
            encode=False,
        )

    def call(self, x, training=False, y=None, adj=None):

        encoded = self.encoder(x=x, adj=adj, state=None)
        decoded = self.decoder(adj=adj, state=encoded, x=y)
        return tf_squeeze(decoded, axis=-1)

    def train_step(self, data):
        adj, x, y = data

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, adj=adj)
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        zipped_gradients = zip(gradients, self.trainable_variables)

        for i, e in zipped_gradients:
            tf.summary.histogram("grads/" + e.name, i)
            tf.summary.scalar("grads/" + e.name + "/max", tf.reduce_max(i))
            tf.summary.scalar("grads/" + e.name + "/min", tf.reduce_min(i))
            tf.summary.scalar("grads/" + e.name + "/mean", tf.reduce_mean(i))

        self.optimizer.apply_gradients(zipped_gradients)

        self.compiled_metrics.update_state(
            {"seq2seq/ar": y}, {"seq2seq/ar": y_pred}, None
        )
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        adj, x, y = data
        y_pred = self(x, training=False, adj=adj)
        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, None, regularization_losses=self.losses)
        self.compiled_metrics.update_state(
            {"seq2seq/ar": y}, {"seq2seq/ar": y_pred}, None
        )
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        adj, x, _ = data
        return self(x, training=False, adj=adj)
