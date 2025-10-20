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
    # Filter the angle which value is lower than 0 or larger than 2*pi
    epsilon = 1e-8
    adjusted_angles = tf.where(vector < 0, vector + 2 * np.pi, vector)
    safe_vector = tf.where(adjusted_angles == 0, epsilon, adjusted_angles)
    return safe_vector


def compute_percentile(y_true, q, keepdims=True):
    # Calculate the threshold which can let the percentile q of the data over this threshold
    reshaped = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    sorted_values = tf.sort(reshaped, axis=1)
    n = tf.shape(sorted_values)[1]

    index = tf.cast(tf.math.ceil(q / 100.0 * tf.cast(n, tf.float32)), tf.int32) - 1
    percentile_value = tf.gather(sorted_values, index, axis=1)

    if keepdims:
        return tf.reshape(percentile_value, [tf.shape(y_true)[0], 1, 1, 1])
    return percentile_value


def convert_abs_back(abs, min, mean, max):
    # Convert the normalization and logarithm back to the original format
    data_abs = abs[:, :, :, :1]
    data_abs = data_abs * (max - min) + min
    data_abs = tf.math.pow(10.0, data_abs)
    return data_abs


def convert_to_complex(data, min, mean, max):
    # Convert the amplitude and phase to complex format
    data_abs = convert_abs_back(data[:, :, :, :1], min, mean, max)

    # angle output is second half of filters
    data_angle = data[:, :, :, 1:] * tf.constant(np.pi, dtype=tf.float32)

    data_real = tf.math.multiply(data_abs, tf.math.cos(data_angle))
    data_imag = tf.math.multiply(data_abs, tf.math.sin(data_angle))

    data_out = tf.complex(data_real, data_imag)

    return data_out


@gin.configurable
def convert_to_time_domain(rD, whether_fftshift):
    # iRDP, convert the frequency domain back time domain
    num_chirps = tf.shape(rD)[2]
    num_samples = tf.shape(rD)[1]
    rD = tf.reshape(rD, (-1, num_samples, num_chirps))

    # reverse Doppler frequency shifting
    if whether_fftshift:
        rD = tf.signal.ifftshift(rD, axes=2)

    # Doppler iFFT
    rfft = tf.signal.ifft(rD)

    range_window = tf.signal.hamming_window((num_samples-1)*2)
    doppler_window = tf.signal.hamming_window(num_chirps)

    # reverse Doppler windowing
    re_rfft = tf.math.real(rfft) / doppler_window
    im_rfft = tf.math.imag(rfft) / doppler_window
    rfft = tf.complex(re_rfft, im_rfft)

    # range iFFT
    rfft = tf.transpose(rfft, perm=[0, 2, 1])
    cube2 = tf.signal.irfft(rfft)

    # reverse range windowing
    cube2 = cube2 / range_window

    return cube2


def loss_function(y_true, y_pred, data_type):
    # MSE loss which compare the prediction and ground truth directly
    if data_type == 'frequency_domain':
        real_loss = tf.reduce_mean(tf.square(y_true[:, :, :, :1] - y_pred[:, :, :, :1]))
        imag_loss = tf.reduce_mean(tf.square(y_true[:, :, :, 1:] - y_pred[:, :, :, 1:]))

        loss = real_loss + 0.001*imag_loss

    elif data_type == 'time_domain':
        loss = tf.reduce_mean(tf.square(convert_to_time_domain(y_true) - convert_to_time_domain(y_pred)))

    else:
        raise ValueError('Data type is not supported.')

    return loss


@gin.configurable
def lsd(y_true, y_pred, data_type, global_min, global_mean, global_max, scaling_factor, thres_type, processing_method):
    # LSD loss function, which applies the logarithm on the amplitude
    processing_type, upsampling_type, separation_type, log_type, abs_normalization_type, angle_normalization_type = processing_method.split('&')
    # LSD mask to filter much noise out
    if thres_type == 'global_mean':
        thres = global_mean
    elif thres_type == 'batch_mean':
        thres = tf.reduce_mean(y_true, axis=[1, 2, 3], keepdims=True)
    elif thres_type == 'batch_median':
        thres = compute_percentile(y_true, q=50.0, keepdims=True)
    else:
        raise ValueError('The thres_type in LSD loss function is not available.')
    thres = thres * scaling_factor

    if separation_type == 'ap/ph' or separation_type == 'ap':
        mask = tf.cast(y_true[:, :, :, :1] > thres, tf.float32)
    elif separation_type == 're/im':
        abs = tf.math.sqrt(tf.square(y_true[:, :, :, :1]) + tf.square(y_true[:, :, :, 1:]))
        mask = tf.cast(abs > thres, tf.float32)
    else:
        raise ValueError('The separation_type in LSD loss function is not available.')
    masked_gt = y_true * mask

    if data_type == 'frequency_domain':
        if separation_type == 'ap/ph' or separation_type == 'ap':
            # If the processing methods contain the logarithm operation, here without the additional logarithm
            if log_type == 'with_log10':
                abs_loss_masked = (tf.square(y_true[:, :, :, :1] - y_pred[:, :, :, :1]))
            else:
                y_true_abs = tf.math.log(safe_log(tf.math.abs(y_true[:, :, :, :1]))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
                y_pred_abs = tf.math.log(safe_log(tf.math.abs(y_pred[:, :, :, :1]))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
                abs_loss_masked = (tf.square(y_true_abs - y_pred_abs))
        elif separation_type == 're/im':
            y_true_abs = tf.math.sqrt(tf.square(y_true[:, :, :, :1]) + tf.square(y_true[:, :, :, 1:]))
            y_pred_abs = tf.math.sqrt(tf.square(y_pred[:, :, :, :1]) + tf.square(y_pred[:, :, :, 1:]))
            abs_loss_masked = (tf.square(y_true_abs - y_pred_abs)) * mask
        else:
            raise ValueError('The separation_type in LSD loss function is not available.')
        abs_loss = tf.math.sqrt(tf.reduce_mean(abs_loss_masked))

        if separation_type == 'ap/ph':
            angle_loss_masked = (tf.square(y_true[:, :, :, 1:] - y_pred[:, :, :, 1:])) * mask
            angle_loss = tf.reduce_mean(angle_loss_masked)
            # The ratio between the amplitude loss and angle loss
            LAMBDA_angle = 1e-1
        elif separation_type == 're/im':
            y_true_angle = tf.math.atan2(y_true[:, :, :, 1:], y_true[:, :, :, :1])
            y_pred_angle = tf.math.atan2(y_pred[:, :, :, 1:], y_pred[:, :, :, :1])
            angle_loss_masked = (tf.square(y_true_angle - y_pred_angle)) * mask
            angle_loss = tf.reduce_mean(angle_loss_masked)
            LAMBDA_angle = 1e-1
        elif separation_type == 'ap':
            angle_loss = 0.0
            LAMBDA_angle = 0.0
        else:
            raise ValueError('The separation_type in LSD loss function is not available.')
        loss = abs_loss + angle_loss * LAMBDA_angle
        return loss, mask, masked_gt
    elif data_type == 'time_domain':
        loss_masked = tf.square(convert_to_time_domain(convert_to_complex(y_true, global_min, global_mean, global_max)) - convert_to_time_domain(convert_to_complex(y_pred, global_min, global_mean, global_max)))
        loss = tf.reduce_mean(loss_masked)
        return loss, mask, masked_gt

def plsd(y_true, y_pred):
    # PLSD loss function, which considers the angle difference in cosine function
    eps = 1e-12  # avoid div by 0
    # loss = tf.reduce_mean(tf.math.multiply(tf.square(tf.experimental.numpy.log10(safe_log(tf.math.abs(y_true))) - tf.experimental.numpy.log10(safe_log(tf.math.abs(y_pred)))),  (2 - tf.math.cos(tf.math.angle(y_pred) - tf.math.angle(y_true)))))
    loss = tf.reduce_mean(tf.math.multiply(tf.square(y_true[:, :, :, :1] - y_pred[:, :, :, :1]),  (2 - tf.math.cos(y_true[:, :, :, 1:] - y_pred[:, :, :, 1:]))))
    return loss


def weighted_mse(y_true, y_pred, min, mean, max, data_type):
    # Apply weights based on the MSE loss
    if data_type == 'frequency_domain':
        # # loss = tf.reduce_mean(tf.math.abs(y_true) * tf.square(tf.math.abs(y_true-y_pred)))
        # abs_loss = tf.reduce_mean(tf.square(convert_abs_back(y_true[:, :, :, :1], min, mean, max) - convert_abs_back(y_pred[:, :, :, :1], min, mean, max)) / convert_abs_back(y_true[:, :, :, :1], min, mean, max))
        # angle_loss = tf.reduce_mean(tf.square(y_true[:, :, :, 1:] - y_pred[:, :, :, 1:]))

        # y_modified = tf.where(y_true[:, :, :, :1] > 0, 1.0, tf.abs(y_true[:, :, :, :1]))
        abs_loss = tf.reduce_mean(tf.square(y_true[:, :, :, :1] - y_pred[:, :, :, :1]) / y_true[:, :, :, :1])
        angle_loss = tf.reduce_mean(tf.square(y_true[:, :, :, 1:] - y_pred[:, :, :, 1:]))

        LAMBDA_angle = 1e-5
        loss = abs_loss + angle_loss * LAMBDA_angle
        return loss

    elif data_type == 'time_domain':
        y_true = convert_to_time_domain(convert_to_complex(y_true, min, mean, max))
        y_pred = convert_to_time_domain(convert_to_complex(y_pred, min, mean, max))
        y_modified = tf.math.log(safe_log(y_true)) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
        y_modified = tf.where(y_modified > 0, 1.0, tf.abs(y_modified))
        loss = tf.reduce_mean(tf.square(y_true-y_pred) / y_modified)
        return loss


def sdr(y_true, y_pred, data_type):
    eps = 1e-12  # avoid div by 0
    # complex signal-to-distortion-ratio SDR: mean(10*log10(|S|^2 / |S - S_hat|^2))
    if data_type == 'frequency_domain':
        diff = y_true - y_pred
        bracket = (tf.math.square(y_true[:, :, :, :1])+eps) / (tf.math.square(diff[:, :, :, :1])+eps)

        SDR = tf.reduce_mean(10. * tf.math.log(bracket) / tf.math.log(tf.constant(10.0, dtype=tf.float32)))
        loss = -SDR

    # SDR in time domain: mean(10*log10(||s||^2 / ||s - s_hat||^2))
    elif data_type == 'time_domain':
        y_true = convert_to_time_domain(y_true)
        y_pred = convert_to_time_domain(y_pred)

        diff = y_true - y_pred
        bracket = tf.reduce_sum(tf.math.square(y_true), axis=[-2, -1]) \
                  / (tf.reduce_sum(tf.math.square(diff), axis=[-2, -1]) + eps)

        # avg over minibatch
        SDR = tf.reduce_mean(10. * tf.math.log(bracket)) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
        loss = - SDR

    else:
        raise ValueError('Data type is not supported.')

    return loss


def sd_sdr(y_true, y_pred, data_type):
    eps = 1e-12  # avoid div by 0
    # Not sure if there is a complex freq. domain definition for sd-sdr
    # https://arxiv.org/pdf/1811.02508 - SNR (eq. 1) in this paper is normal SDR

    # SD-SDR in time domain: same as SDR but with scaling factor alpha = (s_hat^T s) / ||s||^2 (eq. 6 from paper)
    # mean(10*log10(alpha*||s||^2 / ||s - s_hat||^2))
    if data_type == 'time_domain':

        y_true_t = convert_to_time_domain(y_true)
        y_pred_t = convert_to_time_domain(y_pred)
        y_true_vec = tf.reshape(y_true_t, (y_true_t.shape[0], y_true_t.shape[1] * y_true_t.shape[2]))
        y_pred_vec = tf.reshape(y_pred_t, (y_pred_t.shape[0], y_pred_t.shape[1] * y_pred_t.shape[2]))

        alpha = tf.reduce_sum(tf.multiply(y_true_vec, y_pred_vec), axis=-1) \
                / (tf.reduce_sum(tf.math.square(y_true_vec), axis=-1) + eps)

        diff = y_true_t - y_pred_t
        bracket = tf.reduce_sum(tf.math.square(alpha[:, None, None] * y_true_t), axis=[-2, -1]) \
                  / (tf.reduce_sum(tf.math.square(diff), axis=[-2, -1]) + eps)

        SD_SDR = tf.reduce_mean(10. * tf.math.log(bracket)) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
        loss = - SD_SDR

        # # Equivalent implementation (see Eq. 6 from paper linked above)
        # bracket = tf.reduce_sum(tf.math.square(y_true_t), axis=[-2, -1]) \
        #     / (tf.reduce_sum(tf.math.square(diff), axis=[-2, -1]) + eps)

        # SDR = 10. * tf.math.log(bracket) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
        # alpha_term = 20. * tf.math.log(alpha) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

        # SD_SDR = tf.reduce_mean(SDR + alpha_term)
        # loss2 = -SD_SDR

    else:
        raise ValueError('Data type is not supported.')

    return loss


def calc_min_perm_loss(predictions, ground_truth, loss_fct):
    ''' function that calculates the minimum loss of all possible predictions-ground_truth
    permutations.The permutation losses are calculated by fast lookup of the previously
    calculated loss pairs."""

    inputs: - predictions {list}: predicted separated sources, list with C elements (channels)
              of size (B, T) (batch_size, time_steps)
            - ground_truth {list}: true separated sources, similar to predictions
            - loss_fct {string}: which loss function to use; choose from ('mse', 'sd_snr', 'si_snr')


    outputs: - perm_loss: the minimum permutation loss
    '''
    predictions = abs(tf.expand_dims(convert_to_time_domain(predictions), axis=-1))
    ground_truth = abs(tf.expand_dims(convert_to_time_domain(ground_truth), axis=-1))

    # safety for div by zero
    eps = 1e-12

    predictions = tf.reshape(predictions, (predictions.shape[0], predictions.shape[1]*predictions.shape[2], predictions.shape[3]))
    gt = tf.reshape(ground_truth, (ground_truth.shape[0], ground_truth.shape[1] * ground_truth.shape[2], ground_truth.shape[3]))

    B, T, C = predictions.shape

    if loss_fct == 'sd_snr':
        loss = ((tf.reduce_sum(tf.square(tf.expand_dims(tf.einsum('ijk,ijk->ij', predictions, gt)/(tf.reduce_sum(tf.square(gt), axis=-1)), axis=-1) * gt), axis=-1))
                /
                (tf.reduce_sum(tf.square(gt - predictions) + eps, axis=-1)))

        # loss = tf.transpose(tf.convert_to_tensor(loss))
        loss = tf.clip_by_value(loss, clip_value_min=eps, clip_value_max=tf.reduce_max(loss))

        # we want to maximize the SI_SNR which equals minimizing (gradient DESCENT) the
        # negative SI_SNR --> -
        loss = tf.reduce_mean(-tf.math.log(loss) / tf.math.log(tf.constant(10.0, dtype=tf.float32)))

    return loss

@gin.configurable
def generator_loss(loss_function_type, disc_generated_output, gen_output, target, epoch, feature_extractor, global_min, global_mean, global_max, LAMBDA_l1_loss, LAMBDA_perceptual_loss, LAMBDA_gan_loss):
    # The generator loss including LSD, perceptual and adversarial loss
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM) # reduction in default is not fit for distributed training
    # gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    gan_loss = tf.reduce_mean(tf.square(tf.ones_like(disc_generated_output) - disc_generated_output))

    # Mean absolute error
    # l1_loss = tf.reduce_mean(tf.square(convert_to_time_domain(target) - convert_to_time_domain(gen_output)))
    # By default, the LSD is used rather than WMSE here
    if loss_function_type == 'lsd':
        l1_loss, mask, masked_gt = lsd(target, gen_output,'frequency_domain', global_min, global_mean, global_max)
    elif loss_function_type == 'weighted_mse':
        l1_loss = weighted_mse(target, gen_output, global_min, global_mean, global_max, 'frequency_domain')
    else:
        raise ValueError("The loss function is not configured in generator loss function.")

    perceptual_loss = vgg_perceptual_loss(target, gen_output, feature_extractor)

    # For the stability, in the first 50 epochs, only trained by LSD loss
    if epoch < 50:
        LAMBDA_perceptual_loss = 0.0
        LAMBDA_gan_loss = 0.0
    # According to the ration to combine the three loss functions
    total_gen_loss = ((LAMBDA_gan_loss * gan_loss) + (LAMBDA_l1_loss * l1_loss) + (LAMBDA_perceptual_loss * perceptual_loss)) / (LAMBDA_l1_loss + LAMBDA_gan_loss+LAMBDA_perceptual_loss)

    # return total_gen_loss, gan_loss, l1_loss, mask, masked_gt
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    '''The adversarial loss of the discriminator'''
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
    y_true = tf.image.grayscale_to_rgb(y_true[:, :, :, :1])  # (batch, height, width, 3)
    y_pred = tf.image.grayscale_to_rgb(y_pred[:, :, :, :1])  # (batch, height, width, 3)
    y_true = tf.keras.applications.vgg19.preprocess_input(y_true)
    y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred)
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    # Compute perceptual loss across all selected layers
    loss = tf.add_n([tf.reduce_mean(tf.abs(t - p)) for t, p in zip(true_features, pred_features)])
    return loss
