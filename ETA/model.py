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
                    tf.keras.layers.Dense(
                        units=2 * units, activation=tf.keras.layers.LeakyReLU()
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(
                        units=units, activation=tf.keras.layers.LeakyReLU()
                    ),
                ]
            )
            for _ in range(4)
        ]

    def operation(self, x0, support, layer, training=False):
        x = tf.tensordot(support, x0, axes=[1, 1])
        x = tf.transpose(x, [1, 0, 2])

        return layer(x, training=training)

    def call(self, x, support, training=False):

        for i in range(4):
            x += self.operation(
                x, support[i], self.layer[i], training=training
            )

        return x


def calculate_random_walk_matrix(adj_mx):
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = np.diag(d_inv)

    return adj_mx.dot(d_mat_inv).T.dot(d_mat_inv)


class Model(tf_keras.Model):
    def __init__(self):

        super(Model, self).__init__()
        num_nodes = config.model.graph_batch_size
        steps_to_predict = config.model.steps_to_predict

        self.encoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [DCGRUCell(64, 2, num_nodes), DCGRUCell(64, 2, num_nodes)]
            ),
            num_nodes=num_nodes,
            steps_to_predict=steps_to_predict,
        )

        self.decoder = DCGRUBlock(
            tf_keras.layers.StackedRNNCells(
                [
                    DCGRUCell(64, 2, num_nodes),
                    DCGRUCell(64, 2, num_nodes, num_proj=1),
                ]
            ),
            num_nodes=num_nodes,
            steps_to_predict=steps_to_predict,
            encode=False,
        )
        mat = np.load(
            "{}/{}/metr_adj_matrix.npz".format(
                config.model.working_dir, config.model.static_data_dir
            )
        )["arr_0"].astype(np.float32)

        nmat = calculate_random_walk_matrix(mat).T

        adjacency_matrix = [np.eye(len(mat))]

        for _ in range(3):
            adjacency_matrix.append(adjacency_matrix[-1].dot(nmat))

        self.adjacency_matrix = [
            tf.convert_to_tensor(x, dtype=tf.float32) for x in adjacency_matrix
        ]

        self.gconv = GConv(64)
        self.gconv1 = GConv(64)

    def call(self, x, training=False, y=None, adj=None):

        with tf.name_scope("f_call"):

            encoded = self.encoder(x=x, adj=adj, state=None, training=training)

            encoded = [
                self.gconv(
                    encoded[0], self.adjacency_matrix, training=training
                ),
                self.gconv(
                    encoded[1], self.adjacency_matrix, training=training
                ),
            ]
            decoded = self.decoder(
                adj=adj, state=encoded, x=y, training=training
            )
            return tf_squeeze(decoded, axis=-1)

    def train_step(self, data):
        x, y = data

        with tf_diff.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses
            )

        gradients = tape.gradient(loss, self.trainable_variables)

        # for i, e in zip(gradients, self.trainable_variables):
        #     tf.summary.histogram("grads/" + e.name, i)
        #     tf.summary.scalar("grads/" + e.name + "/max", tf.reduce_max(i))
        #     tf.summary.scalar("grads/" + e.name + "/min", tf.reduce_min(i))
        #     tf.summary.scalar("grads/" + e.name + "/mean", tf.reduce_mean(i))

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

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
