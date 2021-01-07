import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
from tensorflow import squeeze as tf_squeeze
from tensorflow.python.keras.engine import data_adapter
from ETA import DCGRUBlock, DCGRUCell, config
import numpy as np


class Model(tf_keras.Model):

    def __init__(self):

        super(Model, self).__init__()
        num_nodes = config.model.graph_batch_size

        self.encoder = DCGRUBlock(tf_keras.layers.StackedRNNCells(
            [DCGRUCell(64, 2, num_nodes) for _ in range(2)]), num_nodes=num_nodes, steps_to_predict=12)
        
        self.decoder = DCGRUBlock(tf_keras.layers.StackedRNNCells([DCGRUCell(64, 2, num_nodes),
                                                                   DCGRUCell(64, 2, num_nodes, num_proj=1)]),
                                  num_nodes=num_nodes, steps_to_predict=12, encode=False)

    def call(self, x, training=False, y=None, adj=None):

        encoded = self.encoder(x=x, adj=adj,  state=None)
        decoded = self.decoder(adj=adj, state=encoded, x=y)
        return tf_squeeze(decoded, axis=-1)
    
    def train_step(self, data):
        adj, x, y = data

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1], adj=adj)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}