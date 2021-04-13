import tensorflow.keras as tf_keras
from ETA import config


class Model(tf_keras.Model):

    def __init__(self):

        super(Model, self).__init__()

        self.network = tf_keras.Sequential([
            tf_keras.layers.Conv1D(filters=64, kernel_size=3, padding="SAME",
                                   activation=tf_keras.layers.LeakyReLU(alpha=0.2)),

            tf_keras.layers.Conv1D(filters=64, kernel_size=3, padding="SAME",
                                   activation=tf_keras.layers.LeakyReLU(alpha=0.2)),

            # Encoder(),

            tf_keras.layers.GRU(units=64, return_sequences=True,
                                return_state=False),
            tf_keras.layers.GRU(units=64, return_sequences=True,
                                return_state=False),

            # Encoder(),

            tf_keras.layers.Conv1D(filters=64, kernel_size=3, padding="SAME",
                                   activation=tf_keras.layers.LeakyReLU(alpha=0.2)),

            tf_keras.layers.Conv1D(filters=config.model.num_nodes, kernel_size=3, padding="SAME",
                                   ),
        ])

    def call(self, x, training=False):
        return self.network(x,
                            training=training)