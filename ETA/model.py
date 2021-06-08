import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
from tensorflow import squeeze as tf_squeeze
from tensorflow.python.keras.engine import data_adapter
from ETA import DCGRUBlock, DCGRUCell, config
import numpy as np


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        adjacency_matrix = np.load(
            "{}/{}/metr_adj_matrix.npz".format(
                config.model.working_dir, config.model.static_data_dir
            )
        )["arr_0"].astype(np.float32)
        num_nodes = config.model.graph_batch_size

        self.encoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [
                    DCGRUCell(64, adjacency_matrix, 2, num_nodes)
                    for _ in range(2)
                ]
            ),
            num_nodes=num_nodes,
            steps_to_predict=12,
        )

        self.decoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [
                    DCGRUCell(64, adjacency_matrix, 2, num_nodes),
                    DCGRUCell(64, adjacency_matrix, 2, num_nodes, num_proj=1),
                ]
            ),
            num_nodes=num_nodes,
            steps_to_predict=12,
            encode=False,
        )

    def call(self, x, pos=None, training=False, y=None):

        encoded = self.encoder(x, state=None, training=training, pos=pos)
        decoded = self.decoder(state=encoded, x=y, training=training, pos=pos)
        return decoded

    def train_step(self, data):
        pos, x, y = data
        sample_weight = None

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1], pos=pos)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        pos, x, y = data
        y_pred = self(x, training=False, pos=pos)
        # Updates stateful loss metrics.
        loss = self.compiled_loss(
            y, y_pred, None, regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)
