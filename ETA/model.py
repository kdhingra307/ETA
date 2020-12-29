import tensorflow.keras as tf_keras
import tensorflow.autodiff as tf_diff
from tensorflow import squeeze as tf_squeeze
from tensorflow.python.keras.engine import data_adapter
from ETA import DCGRUBlock, DCGRUCell, config
import numpy as np


class Encoder(tf_keras.layers.Layer):

    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = [
            tf_keras.Sequential([
                tf_keras.layers.Conv1D(filters=256, kernel_size=3,
                                       padding="SAME", activation=tf_keras.layers.LeakyReLU(alpha=0.2)),
                tf_keras.layers.BatchNormalization()])
        ]*3

    def call(self, x, training=False):
        for each_layer in self.layers:
            x += each_layer(x, training)
        return x


class Model(tf_keras.Model):

    def __init__(self):

        super(Model, self).__init__()

        adjacency_matrix = np.load("{}/{}/metr_adj_matrix.npz".format(config.model.working_dir, config.model.static_data_dir))['arr_0'].astype(np.float32)
        num_nodes = config.model.num_nodes

        self.encoder = DCGRUBlock(tf_keras.layers.StackedRNNCells(
            [DCGRUCell(64, adjacency_matrix, 2, num_nodes) for _ in range(2)]), num_nodes=num_nodes, steps_to_predict=6)
        
        self.decoder = DCGRUBlock(tf_keras.layers.StackedRNNCells([DCGRUCell(64, adjacency_matrix, 2, num_nodes),
                                                                   DCGRUCell(64, adjacency_matrix, 2, num_nodes, num_proj=1)]),
                                  num_nodes=num_nodes, steps_to_predict=6, encode=False)

    def call(self, x, training=False, y=None):

        encoded = self.encoder(x, state=None)
        decoded = self.decoder(state=encoded, x=y)
        return tf_squeeze(decoded, axis=-1)
    
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True, y=y[:, :, :, :1])
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}