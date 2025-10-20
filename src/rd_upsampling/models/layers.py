import tensorflow as tf


def upsampling_layer(input):
    """
    A simple upsampling layer
    """

    output = tf.keras.layers.UpSampling2D()(input)

    return output


class ReImSeparation(tf.keras.layers.Layer):
    def call(self, input):
        real_part = tf.math.real(input)
        imaginary_part = tf.math.imag(input)

        output = tf.concat([real_part, imaginary_part], axis=-1)
        return output


def re_im_separation(input):
    real_part = tf.math.real(input)
    imaginary_part = tf.math.imag(input)

    output = tf.concat([real_part, imaginary_part], axis=-1)

    return output


class amplitude_phase_separation(tf.keras.layers.Layer):
    def __init__(self, log_type):
        super(amplitude_phase_separation, self).__init__()
        self.log_type = log_type

    def call(self, inputs):
        inputs_abs = tf.math.abs(inputs)
        inputs_angle = tf.math.angle(inputs)

        if self.log_type == 'with_log':
            inputs_abs = tf.clip_by_value(inputs_abs, clip_value_min=1e-25, clip_value_max=tf.reduce_max(inputs_abs))
            # inputs_abs = tf.experimental.numpy.log10(inputs_abs)
            inputs_abs = tf.math.log(inputs_abs) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

        # # normalisation with the standard deviation
        # stds = tf.math.reduce_std(cube, keepdims=True)
        #
        # # normalize the fast-time samples per chirp by the standard deviation
        # cube = cube / stds

        # normalisation with the maximum value
        # for float case
        maxs = tf.math.reduce_max(tf.math.abs(inputs_abs), keepdims=True)
        mean = tf.math.reduce_mean(inputs_abs, axis=[1, 2], keepdims=True)
        inputs_abs = (inputs_abs - mean) / maxs

        # inputs_exponential = tf.concat([inputs_abs, inputs_angle], axis=-1)
        inputs_exponential = tf.concat([inputs_abs], axis=-1)

        # inputs_abs_expanded = tf.expand_dims(inputs_abs, axis=-1)
        # inputs_angle_expanded = tf.expand_dims(inputs_angle, axis=-1)
        #
        # combined = tf.concat([inputs_abs_expanded, inputs_angle_expanded], axis=-1)  # (batch_size, ..., frame_length, 2)
        #
        # inputs_exponential = tf.reshape(tf.transpose(combined, perm=[0, 1, 2, 4, 3]), shape=(-1, combined.shape[1], combined.shape[2], combined.shape[-1]*combined.shape[-2]))

        return inputs_exponential, tf.expand_dims(mean[:, :, :, -1], axis=-1), maxs


class amplitude_phase_postprocessing(tf.keras.layers.Layer):
    def call(self, out, mean, min, max, log_type):
        out_abs = out[:, :, :, :1]

        out_abs = out_abs * (max-min) + min

        if log_type == 'with_log':
            out_abs = tf.math.pow(10.0, out_abs)
            # out_abs = out_abs

        # angle output is second half of filters
        out_angle = out[:, :, :, 1:]

        out_real = tf.math.multiply(out_abs, tf.math.cos(out_angle))
        out_imag = tf.math.multiply(out_abs, tf.math.sin(out_angle))

        out = tf.complex(out_real, out_imag)

        return out