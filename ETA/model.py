import tensorflow.keras as tf_keras


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

        self.network = tf_keras.Sequential([
            tf_keras.layers.Conv1D(filters=2048, kernel_size=3, padding="SAME",
                                   activation=tf_keras.layers.LeakyReLU(alpha=0.2)),

            tf_keras.layers.Conv1D(filters=256, kernel_size=3, padding="SAME",
                                   activation=tf_keras.layers.LeakyReLU(alpha=0.2)),

            # Encoder(),

            tf_keras.layers.GRU(units=512, return_sequences=True,
                                return_state=False),
            tf_keras.layers.GRU(units=512, return_sequences=True,
                                return_state=False),

            # Encoder(),

            tf_keras.layers.Conv1D(filters=2048, kernel_size=3, padding="SAME",
                                   activation=tf_keras.layers.LeakyReLU(alpha=0.2)),

            tf_keras.layers.Conv1D(filters=6001, kernel_size=3, padding="SAME",
                                   activation=tf_keras.layers.ReLU()),
        ])

    def call(self, x, training=False):
        return self.network(x,
                            training=training)