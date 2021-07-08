import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
from tensorflow import squeeze as tf_squeeze
from tensorflow.python.keras.engine import data_adapter
from ETA import DCGRUBlock, DCGRUCell, config
import numpy as np
import tensorflow as tf
from ETA.grud import GRUDCell


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        adjacency_matrix = None

        num_nodes = config.model.graph_batch_size

        cell = GRUDCell(256, 2, num_nodes)
        cell.build([None, None, None, 2])
        self.encoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [
                    cell,
                    DCGRUCell(256, adjacency_matrix, 2, num_nodes),
                ]
            ),
            num_nodes=num_nodes,
            steps_to_predict=config.model.steps_to_predict,
        )

        self.decoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [
                    DCGRUCell(256, adjacency_matrix, 2, num_nodes),
                    DCGRUCell(256, adjacency_matrix, 2, num_nodes, num_proj=1),
                ]
            ),
            num_nodes=num_nodes,
            steps_to_predict=config.model.steps_to_predict,
            encode=False,
        )

    def call(self, x, pos=None, training=False, y=None, z=None):

        encoded = self.encoder(x, state=None, training=training, pos=pos, z=z)
        decoded = self.decoder(
            state=encoded, x=y, training=training, pos=pos, z=z
        )
        return decoded

    def train_step(self, data):
        pos, x, y, z = data

        sample_weight = None

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1], pos=pos, z=z)
            Lreg = 0.0015 * sum(
                tf.nn.l2_loss(tf_var) for tf_var in self.trainable_variables
            )
            loss = (
                self.compiled_loss(
                    y, y_pred, None, regularization_losses=self.losses
                )
                + Lreg
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        pos, x, y, z = data
        y_pred = self(x, training=False, pos=pos, z=z)
        # Updates stateful loss metrics.
        loss = self.compiled_loss(
            y, y_pred, None, regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
