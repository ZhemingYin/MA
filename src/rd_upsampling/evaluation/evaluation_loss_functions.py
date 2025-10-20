import tensorflow as tf
import gin
import numpy as np
import wandb
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

from evaluation.visualization import plot_range_doppler


def safe_log(vector):
    # Make sure that no zeros are in the vector that cause nan problem in log
    epsilon = tf.constant(1e-8, dtype=vector.dtype)
    safe_vector = tf.where(vector == 0, epsilon, vector)
    return safe_vector


def safe_angle_log(vector):
    epsilon = 1e-8
    adjusted_angles = tf.where(vector < 0, vector + 2 * np.pi, vector)
    safe_vector = tf.where(adjusted_angles == 0, epsilon, adjusted_angles)
    return safe_vector


def compute_percentile(y_true, q, keepdims=True):
    reshaped = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    sorted_values = tf.sort(reshaped, axis=1)
    n = tf.shape(sorted_values)[1]

    index = tf.cast(tf.math.ceil(q / 100.0 * tf.cast(n, tf.float32)), tf.int32) - 1
    percentile_value = tf.gather(sorted_values, index, axis=1)

    if keepdims:
        return tf.reshape(percentile_value, [tf.shape(y_true)[0], 1, 1, 1])
    return percentile_value


def convert_abs_back(abs, min, mean, max):
    data_abs = abs[:, :, :, :1]
    data_abs = data_abs * (max - min) + min
    data_abs = tf.math.pow(10.0, data_abs)
    return data_abs


def convert_to_complex(data, min, mean, max):
    data_abs = convert_abs_back(data[:, :, :, :1], min, mean, max)

    # angle output is second half of filters
    data_angle = data[:, :, :, 1:] * tf.constant(np.pi, dtype=tf.float32)

    data_real = tf.math.multiply(data_abs, tf.math.cos(data_angle))
    data_imag = tf.math.multiply(data_abs, tf.math.sin(data_angle))

    data_out = tf.complex(data_real, data_imag)

    return data_out


def loss_function(y_true, y_pred):
    amp_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return amp_loss


def lsd(y_true, y_pred):
    y_true_log10 = tf.math.log(safe_log(y_true)) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    y_pred_log10 = tf.math.log(safe_log(y_pred)) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    abs_loss_masked = (tf.square(y_true_log10 - y_pred_log10))
    abs_loss = tf.math.sqrt(tf.reduce_mean(abs_loss_masked))
    return abs_loss


def weighted_mse(y_true, y_pred):
    y_true_log10 = tf.math.log(safe_log(tf.math.abs(y_true))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    y_pred_log10 = tf.math.log(safe_log(tf.math.abs(y_pred))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

    # y_modified = tf.where(y_true_log10[:, :, :, :1] > 0, 1.0, tf.abs(y_true_log10[:, :, :, :1]))
    abs_loss = tf.reduce_mean(tf.square(y_true - y_pred) / safe_log(y_true))

    return abs_loss


def sdr(y_true, y_pred):
    eps = 1e-12  # avoid div by 0
    # complex signal-to-distortion-ratio SDR: mean(10*log10(|S|^2 / |S - S_hat|^2))

    diff = y_true - y_pred
    bracket = tf.math.square(y_true) / (tf.math.square(diff) + eps)

    SDR = tf.reduce_mean(10. * tf.math.log(safe_log(bracket)) / tf.math.log(tf.constant(10.0, dtype=tf.float32)))
    loss = -SDR

    return loss


def generator_loss(loss_function_type, disc_generated_output, gen_output, target, epoch, feature_extractor):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM) # reduction in default is not fit for distributed training
    # gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    gan_loss = tf.reduce_mean(tf.square(tf.ones_like(disc_generated_output) - disc_generated_output))

    # Mean absolute error
    # l1_loss = tf.reduce_mean(tf.square(convert_to_time_domain(target) - convert_to_time_domain(gen_output)))
    if loss_function_type == 'lsd':
        l1_loss, mask, masked_gt = lsd(target, gen_output, data_type=None, global_mean=None, scaling_factor=1, thres_type='batch_median')
    elif loss_function_type == 'perceptual':
        l1_loss = vgg_perceptual_loss(target, gen_output, feature_extractor)
    else:
        raise ValueError("The loss function is not configured in generator loss function.")

    if epoch < 50:
        LAMBDA_l1_loss = 1.0
        LAMBDA_gan_loss = 0.0
    else:
        LAMBDA_l1_loss = 1.0
        LAMBDA_gan_loss = 0.01

    total_gen_loss = ((LAMBDA_gan_loss * gan_loss) + (LAMBDA_l1_loss * l1_loss)) / (LAMBDA_l1_loss + LAMBDA_gan_loss)

    # return total_gen_loss, gan_loss, l1_loss, mask, masked_gt
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    # real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    real_loss = tf.reduce_mean(tf.square(tf.ones_like(disc_real_output) - disc_real_output))

    # generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    generated_loss = tf.reduce_mean(tf.square(tf.zeros_like(disc_generated_output) - disc_generated_output))

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def vgg_perceptual_loss(y_true, y_pred, feature_extractor):
    """
    Computes the perceptual loss between two images using VGG19.
    Args:
        y_true (tensor): Ground truth image.
        y_pred (tensor): Predicted image.
        layer_names (list of str): The VGG19 layers to extract features from.
    Returns:
        Perceptual loss value.
    """
    # Expand the feature of rD map, turn to RGB rather than gray image
    y_true = tf.image.grayscale_to_rgb(y_true)  # (batch, height, width, 3)
    y_pred = tf.image.grayscale_to_rgb(y_pred)  # (batch, height, width, 3)
    y_true = tf.keras.applications.vgg19.preprocess_input(y_true)
    y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred)
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    # Compute perceptual loss across all selected layers
    loss = tf.add_n([tf.reduce_mean(tf.abs(t - p)) for t, p in zip(true_features, pred_features)])
    return loss