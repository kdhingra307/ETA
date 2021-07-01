import tensorflow as tf
from tensorflow.python.keras import backend as K


class GRUCell(tf.keras.layers.GRUCell):
    def __init__(self, units):
        super(GRUCell, self).__init__(units)
        self.x_prev = tf.keras.layers.Dense(208, name="x_prev")
        self.h_prev = tf.keras.layers.Dense(units, name="h_prev")
        self.mask_z = tf.keras.layers.Dense(
            units, name="mask_z", use_bias=False
        )
        self.mask_r = tf.keras.layers.Dense(
            units, name="mask_r", use_bias=False
        )
        self.mask_h = tf.keras.layers.Dense(
            units, name="mask_h", use_bias=False
        )

    def call(self, inputs, states, training=False, constants=None):
        x2 = constants[0][:, 0]

        x = inputs[:, :208]
        x1 = inputs[:, 208 : 2 * 208]
        mask = inputs[:, 2 * 208 : 3 * 208]
        dt = inputs[:, 3 * 208 :]

        mask = mask[:, :1]
        dt = dt[:, :1]

        x_prev_mask = self.x_prev(dt)

        x_prev_mask = tf.exp(
            -1 * tf.clip_by_value(x_prev_mask, 0, tf.float32.max)
        )

        inputs = (x * mask) + (
            (1 - mask) * (x_prev_mask * x1 + (1 - x_prev_mask) * x2)
        )

        h_prev_mask = self.h_prev(dt)

        h_prev_mask = tf.exp(
            -1 * tf.clip_by_value(h_prev_mask, 0, tf.float32.max)
        )

        h_tm1 = h_prev_mask * states[0]

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = tf.unstack(self.bias)

        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

        x_z = K.dot(inputs_z, self.kernel[:, : self.units])
        x_r = K.dot(inputs_r, self.kernel[:, self.units : self.units * 2])
        x_h = K.dot(inputs_h, self.kernel[:, self.units * 2 :])

        if self.use_bias:
            x_z = K.bias_add(x_z, input_bias[: self.units])
            x_r = K.bias_add(x_r, input_bias[self.units : self.units * 2])
            x_h = K.bias_add(x_h, input_bias[self.units * 2 :])

        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

        recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, : self.units])
        recurrent_r = K.dot(
            h_tm1_r, self.recurrent_kernel[:, self.units : self.units * 2]
        )

        if self.reset_after and self.use_bias:
            recurrent_z = K.bias_add(recurrent_z, recurrent_bias[: self.units])
            recurrent_r = K.bias_add(
                recurrent_r, recurrent_bias[self.units : self.units * 2]
            )

        z = self.recurrent_activation(x_z + recurrent_z + self.mask_z(mask))
        r = self.recurrent_activation(x_r + recurrent_r + self.mask_r(mask))

        # reset gate applied after/before matrix multiplication
        if self.reset_after:
            recurrent_h = K.dot(
                h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
            )
            if self.use_bias:
                recurrent_h = K.bias_add(
                    recurrent_h, recurrent_bias[self.units * 2 :]
                )
            recurrent_h = r * recurrent_h
        else:
            recurrent_h = K.dot(
                r * h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
            )

        hh = self.activation(x_h + recurrent_h + self.mask_h(mask))

        h = z * h_tm1 + (1 - z) * hh
        new_state = [h]
        return h, new_state
