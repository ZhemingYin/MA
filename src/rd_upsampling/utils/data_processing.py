import tensorflow as tf
import gin
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tensorflow.python.layers.normalization import normalization


def safe_log(vector):
    # Make sure that no zeros are in the vector that cause nan problem in log
    epsilon = tf.constant(1e-8, dtype=vector.dtype)
    safe_vector = tf.where(vector == 0, epsilon, vector)
    return safe_vector


def range_fft(cube, range_window, data_type, data_size_info):
    """
    Performs range fft for real-valued (only In-phase channel) baseband signal.
    This preserves signal energy (Parseval theorem) such that the windowed
    time-domain cube has the same energy as its fft (range-fft) result.

    Parameters
    ----------
    cube : float
        A 2D (one antenna) baseband data cube, i.e. float array of shape
        (frame length, slow time samples, fast time samples).

    range_window : scipy.signal.windows instance
        A window function to window the time-domain samples before fft.

    Returns
    -------
    r_fft : complex
        The range fft output for each chirp. We use an even number of samples
        and have Hermitian symmetry in the spectrum
        --> shape (frame length, slow time samples, fast time samples)
        --> type of tf.complex64
    """
    if data_type == 'high_res':
        data_size_info[1] = 1
        data_size_info[0] = 1
    num_frames = data_size_info[0]
    num_chirps = data_size_info[3] // data_size_info[1]
    num_samples = data_size_info[2] // data_size_info[1]

    # window the time-domain data
    cube = cube * range_window

    # Compute the FFT (range fft along last axis) of the real-valued (I-channel) baseband signal
    # Convert cube to complex64
    cube = tf.cast(cube, tf.complex64)
    # Do FFT only on the range dimension
    r_fft = tf.signal.fft(cube)

    # normalize by sqrt(N) such that we get factor 1/N in power after squaring this later.
    # In fft we sum over N samples, so we need the factor 1/N to normalize in frequency domain after fft.
    scaling_factor = 1 / np.sqrt(num_samples)
    r_fft = r_fft * scaling_factor

    return r_fft


def range_Doppler(cube, data_type, data_size_info):
    """
    Computes the complex-valued range-Doppler map (2D fft of baseband signal --> still ~ voltage/amplitude U).
    rD input for CFAR is abs(rD)² ~ spectral power P ~ U² : square law detector

    Like in the range_fft() function, the energy of the windowed data before
    and after fft is preserved.

    Parameters
    ----------
    cube : np.ndarray
        A 2D (one antenna) baseband data cube, i.e. float array of shape
        (frame length, slow time samples, fast time samples).
    data_size_info : list of [frame_length, sampling_rate, num_samples, num_chirps]

    Returns
    -------
    rD : np.ndarray
        A complex valued range-Doppler signal of shape
        (frame length, fast time samples, slow time samples)

    """
    if data_type == 'high_res':
        data_size_info[1] = 1
        data_size_info[0] = 1
    num_frames = data_size_info[0]
    num_chirps = data_size_info[3] // data_size_info[1]
    num_samples = data_size_info[2] // data_size_info[1]

    # compute row (chirp) averages
    avgs = tf.reduce_mean(cube, axis=2, keepdims=True)

    # de-bias values --> the fast-time samples per chirp have zero mean
    cube = cube - avgs

    # compute Blackman-Harris window matrix for range processing (fast-time)
    range_window = signal.windows.blackmanharris(num_samples).reshape(1, num_samples)
    # Convert the range_window to tensor
    range_window = tf.convert_to_tensor(range_window, dtype=tf.float32)
    range_window = tf.reshape(range_window, (1, 1, num_samples))

    # compute Blackman-Harris window matrix for Doppler processing (slow-time)
    doppler_window = signal.windows.blackmanharris(num_chirps).reshape(1, num_chirps)
    # Convert the doppler_window to tensor, according to r_fft after range_fft, the type should be complex64
    doppler_window = tf.convert_to_tensor(doppler_window, dtype=tf.complex64)
    doppler_window = tf.reshape(doppler_window, (1, 1, num_chirps))

    # perform range fft for all chirps (fast-time)
    r_fft = range_fft(cube, range_window, data_type, data_size_info)

    # switch axes for Doppler fft
    r_fft = tf.transpose(r_fft, perm=[0, 2, 1])

    # apply window for range-processed Doppler samples
    r_fft = tf.multiply(r_fft, doppler_window)

    # Doppler fft across all chirps (slow-time) - this has complex-valued inputs --> normal fft
    r_fft = tf.cast(r_fft, tf.complex64)
    rD_fft = tf.signal.fft(r_fft) / np.sqrt(num_chirps)

    # shift frequencies such that zero velocity/Doppler is in center
    rD_fft = tf.signal.fftshift(rD_fft, 2)

    return rD_fft


@gin.configurable
def processing(input, data_type, processing_type, logger, data_size_info, R=50):
    output = range_Doppler(input, data_type, data_size_info)
    if processing_type == 'voltage':
        output = output
    elif processing_type == 'power':
        output = tf.abs(output)**2 / R
    elif processing_type == 'dBW':
        output = 10 * tf.math.log(output) / tf.math.log(10.0)
    elif processing_type == 'dBm':
        output = 10 * tf.math.log(output) / tf.math.log(10.0) + 30
    else:
        logger.error("The processing type is not supported.")

    return output


@gin.configurable
def resampling_processing(cube, resampling_factor, num_chirps, num_samples):
    cube_downsampled = cube[:, :num_chirps // resampling_factor, :num_samples // resampling_factor]
    return cube_downsampled


def normalisation_processing(low_res, high_res, global_max, global_mean, global_min, processing_method):
    processing_type, upsampling_type, separation_type, log_type, abs_normalization_type, angle_normalization_type = processing_method.split('&')

    if separation_type == 'ap/ph':
        if log_type == 'with_log10':
            low_res_abs = tf.math.log(safe_log(tf.math.abs(low_res))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
            high_res_abs = tf.math.log(safe_log(tf.math.abs(high_res))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
        elif log_type == 'with_log2':
            low_res_abs = tf.math.log(safe_log(tf.math.abs(low_res))) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
            high_res_abs = tf.math.log(safe_log(tf.math.abs(high_res))) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
        elif log_type == 'no_log':
            low_res_abs = tf.math.abs(low_res)
            high_res_abs = tf.math.abs(high_res)
        else:
            raise ValueError("The log type is not supported in normalization_processing function.")

        # maxs = tf.math.reduce_max(tf.math.abs(low_res_abs), keepdims=True)
        maxs = global_max
        # mean = tf.math.reduce_mean(low_res_abs, axis=[1, 2], keepdims=True)
        mean = global_mean
        min = global_min

        if abs_normalization_type == 'abs_normalization(0,1)':
            low_res_abs_normalised = (low_res_abs - min) / (maxs - min)
            # high_res_abs_normalised = (high_res_abs - tf.expand_dims(mean[:, :, :, -1], axis=-1)) / (maxs+epsilon)
            high_res_abs_normalised = (high_res_abs - min) / (maxs - min)
        elif abs_normalization_type == 'abs_normalization(-1,1)':
            low_res_abs_normalised = low_res_abs / maxs
            high_res_abs_normalised = high_res_abs  / maxs
        elif abs_normalization_type == 'no_abs_normalization':
            low_res_abs_normalised = low_res_abs
            high_res_abs_normalised = high_res_abs
        else:
            raise ValueError("The normalization type is not supported in normalization_processing function.")

        if angle_normalization_type == 'angle_normalization':
            low_res_angle = tf.math.angle(low_res) / tf.constant(np.pi, dtype=tf.float32)
            high_res_angle = tf.math.angle(high_res) / tf.constant(np.pi, dtype=tf.float32)
        elif angle_normalization_type == 'no_angle_normalization':
            low_res_angle = tf.math.angle(low_res)
            high_res_angle = tf.math.angle(high_res)
        else:
            raise ValueError("The angle normalization type is not supported in normalization_processing function.")

        low_res_abs_normalised = tf.concat([low_res_abs_normalised, low_res_angle], axis=-1)
        high_res_abs_normalised = tf.concat([high_res_abs_normalised, high_res_angle], axis=-1)

        return low_res_abs_normalised, high_res_abs_normalised, mean, maxs, min

    elif separation_type == 'ap':
        if log_type == 'with_log10':
            low_res_abs = tf.math.log(safe_log(tf.math.abs(low_res))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
            high_res_abs = tf.math.log(safe_log(tf.math.abs(high_res))) / tf.math.log(
                tf.constant(10.0, dtype=tf.float32))
        elif log_type == 'with_log2':
            low_res_abs = tf.math.log(safe_log(tf.math.abs(low_res))) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
            high_res_abs = tf.math.log(safe_log(tf.math.abs(high_res))) / tf.math.log(
                tf.constant(2.0, dtype=tf.float32))
        elif log_type == 'no_log':
            low_res_abs = tf.math.abs(low_res)
            high_res_abs = tf.math.abs(high_res)
        else:
            raise ValueError("The log type is not supported in normalization_processing function.")

        maxs = global_max
        mean = global_mean
        min = global_min

        if abs_normalization_type == 'abs_normalization(0,1)':
            low_res_abs_normalised = (low_res_abs - min) / (maxs - min)
            # high_res_abs_normalised = (high_res_abs - tf.expand_dims(mean[:, :, :, -1], axis=-1)) / (maxs+epsilon)
            high_res_abs_normalised = (high_res_abs - min) / (maxs - min)
        elif abs_normalization_type == 'abs_normalization(-1,1)':
            low_res_abs_normalised = low_res_abs / maxs
            high_res_abs_normalised = high_res_abs / maxs
        elif abs_normalization_type == 'no_abs_normalization':
            low_res_abs_normalised = low_res_abs
            high_res_abs_normalised = high_res_abs
        else:
            raise ValueError("The normalization type is not supported in normalization_processing function.")

        return low_res_abs_normalised, high_res_abs_normalised, mean, maxs, min

    elif separation_type == 're/im':
        low_res_real_part = tf.math.real(low_res)
        low_res_imaginary_part = tf.math.imag(low_res)
        low_res_input = tf.concat([low_res_real_part, low_res_imaginary_part], axis=-1)

        high_res_real_part = tf.math.real(high_res)
        high_res_imaginary_part = tf.math.imag(high_res)
        high_res_input = tf.concat([high_res_real_part, high_res_imaginary_part], axis=-1)

        maxs = global_max
        mean = global_mean
        min = global_min

        return low_res_input, high_res_input, mean, maxs, min


def normalisation_back(super_res, max, mean, min, processing_method):
    # super_res_unnormalised = super_res * (max - min) + min
    # out = tf.math.pow(10.0, super_res_unnormalised)

    processing_type, upsampling_type, separation_type, log_type, abs_normalization_type, angle_normalization_type = processing_method.split('&')

    if separation_type == 'ap/ph':
        if abs_normalization_type == 'abs_normalization(0,1)':
            super_res_abs = super_res[:, :, :, :1] * (max - min) + min
        elif abs_normalization_type == 'abs_normalization(-1,1)':
            super_res_abs = super_res[:, :, :, :1] * max
        elif abs_normalization_type == 'no_abs_normalization':
            super_res_abs = super_res[:, :, :, :1]
        else:
            raise ValueError("The separation type is not supported in normalization_back function.")

        if log_type == 'with_log10':
            super_res_abs = tf.math.pow(10.0, super_res_abs)
        elif log_type == 'with_log2':
            super_res_abs = tf.math.pow(2.0, super_res_abs)
        elif log_type == 'no_log':
            super_res_abs = super_res_abs
        else:
            raise ValueError("The log type is not supported in normalization_back function.")

        if angle_normalization_type == 'angle_normalization':
            super_res_angle = super_res[:, :, :, 1:] * tf.constant(np.pi, dtype=tf.float32)
        elif angle_normalization_type == 'no_angle_normalization':
            super_res_angle = super_res[:, :, :, 1:]
        else:
            raise ValueError("The angle normalization type is not supported in normalization_back function.")

        out_real = tf.math.multiply(super_res_abs, tf.math.cos(super_res_angle))
        out_imag = tf.math.multiply(super_res_abs, tf.math.sin(super_res_angle))
        out = tf.complex(out_real, out_imag)

        return out

    elif separation_type == 'ap':
        if abs_normalization_type == 'abs_normalization(0,1)':
            super_res_abs = super_res * (max - min) + min
        elif abs_normalization_type == 'abs_normalization(-1,1)':
            super_res_abs = super_res * max
        elif abs_normalization_type == 'no_abs_normalization':
            super_res_abs = super_res
        else:
            raise ValueError("The separation type is not supported in normalization_back function.")

        if log_type == 'with_log10':
            out = tf.math.pow(10.0, super_res_abs)
        elif log_type == 'with_log2':
            out = tf.math.pow(2.0, super_res_abs)
        elif log_type == 'no_log':
            out = super_res_abs
        else:
            raise ValueError("The log type is not supported in normalization_back function.")

        return out

    elif separation_type == 're/im':
        super_res_real_part = super_res[:, :, :, :1]
        super_res_imaginary_part = super_res[:, :, :, 1:]
        super_res_output = tf.complex(super_res_real_part, super_res_imaginary_part)

        return super_res_output

    else:
        raise ValueError("The separation type is not supported in normalization_back function.")


@gin.configurable
def rD_processing(cube, whether_fftshift):
    # Tensorflow only has "hann" and "hamming" windows built in. scipy has some more.
    # I think "hamming" is actually the better window (I also switched to "hamming" in the numpy code).
    # This will give a sharper rD map (hamming window has smaller main lobe in frequency domain than blackmanharris)
    range_window = tf.signal.hamming_window(tf.shape(cube)[2])
    # range_window = signal.windows.blackmanharris(cube.shape[1]).astype(np.float32)
    # range_window = tf.convert_to_tensor(range_window)
    doppler_window = tf.signal.hamming_window(tf.shape(cube)[1])
    # doppler_window = signal.windows.blackmanharris(cube.shape[0]).astype(np.float32)
    # doppler_window = tf.convert_to_tensor(doppler_window, dtype=tf.complex64)

    # windowing for range
    cube_windowed = cube * range_window

    # range fft
    rfft = tf.signal.rfft(cube_windowed)

    # windowing for Doppler
    rfft = tf.transpose(rfft, perm=[0, 2, 1])
    # plt.plot(tf.math.real(rfft[-1, 1, :]), color='red')
    # wandb.log({'Real part after windowing for Doppler': wandb.Image(plt)})
    # plt.close('all')
    # plt.plot(tf.math.imag(rfft[-1, 1, :]), color='orangered')
    # wandb.log({'Imaginary part after windowing for Doppler': wandb.Image(plt)})
    # plt.close('all')

    re_rfft = tf.math.real(rfft) * doppler_window
    im_rfft = tf.math.imag(rfft) * doppler_window
    rfft = tf.complex(re_rfft, im_rfft)

    # Doppler fft and centering of Doppler (0 Doppler in middle)
    # rD is the same as before (complex rD map ~ voltage) but without all the
    # scaling. This goes to the first trainable layer (or some preprocessing before that).
    rD = tf.signal.fft(rfft)
    if whether_fftshift:
        rD = tf.signal.fftshift(rD, 2)

    # # plot rD result
    # plt.figure()
    # plt.pcolormesh(10 * tf.experimental.numpy.log10(tf.math.abs(rD[-1, :, :]) ** 2).numpy())
    # wandb.log({"Range-Doppler processing result": wandb.Image(plt)})

    # plt.close('all')

    return rD

