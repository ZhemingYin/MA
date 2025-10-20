import gin
import tensorflow as tf
import wandb
import math

from models.layers import re_im_separation, amplitude_phase_separation, amplitude_phase_postprocessing


@gin.configurable
def interpolation_upsampling(input_size_info, processing_method):
    '''Interpolation model'''
    processing_type, upsampling_type, separation_type, log_type, abs_normalization_type, angle_normalization_type = processing_method.split('&')

    if separation_type == 'ap/ph' or separation_type == 're/im':
        input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2] * 2)
        inputs = tf.keras.Input(shape=input_size, dtype=tf.float32)
        output_0 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, input_size_info[2]-1], axis=-1))(inputs)
        output_1 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, -1], axis=-1))(inputs)
        scale = math.log2(input_size_info[1])
        for i in range(int(scale)):
            output_0 = tf.image.resize(output_0, [tf.shape(output_0)[1] * 2, tf.shape(output_0)[2] * 2], method='bilinear')
            output_1 = tf.image.resize(output_1, [tf.shape(output_1)[1] * 2, tf.shape(output_1)[2] * 2], method='bilinear')

        output_0 = output_0[:, :-(input_size_info[1] - 1), :, :]
        output_1 = output_1[:, :-(input_size_info[1] - 1), :, :]
        output = tf.concat([output_0, output_1], axis=-1)

    elif separation_type == 'ap':
        input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2])
        inputs = tf.keras.Input(shape=input_size, dtype=tf.float32)
        output = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, -1], axis=-1))(inputs)
        scale = math.log2(input_size_info[1])
        for i in range(int(scale)):
            output = tf.image.resize(output, [tf.shape(output)[1] * 2, tf.shape(output)[2] * 2], method='bilinear')
        output = output[:, :-(input_size_info[1] - 1), :, :]

    else:
        raise ValueError('The separation type is not supported in unet_concat function.')

    return tf.keras.Model(inputs=inputs, outputs=output, name='Interpolation')


@gin.configurable
def CNN_simple(input_size_info, neuron_list, upsampling_layer_idx, processing_method, kernel_size, stride, sampling_rate):

    """
    Define a simple CNN model
    Args:
        input_size_info: the info about the input size, in the format of [batch_size, sampling_rate, frame_length, num_samples, num_chirps]
        neuron_list: the number of neurons as a list which will be used for this model
        upsampling_layer_idx: the index of the layer after which the upsampling layer will be used, starting from 0
    Return:
        keras model object
    """

    # set the input
    processing_type, upsampling_type, separation_type, log_type, abs_normalization_type, angle_normalization_type = processing_method.split('&')

    if separation_type == 'ap/ph' or separation_type == 're/im':
        input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2] * 2)
        last_unit = 2
    elif separation_type == 'ap':
        input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2])
        last_unit = 1
    else:
        raise ValueError('The separation type is not supported in unet_concat function.')
    inputs = tf.keras.Input(shape=input_size, dtype=tf.float32)

    if processing_type == 'padding':
        paddings = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]])
        output = tf.pad(inputs, paddings, mode='CONSTANT', constant_values=0)
    elif processing_type == 'conv':
        output = tf.keras.layers.Conv2D(filters=8, kernel_size=(2, 1), strides=(1, 1), padding='valid')(inputs)
    elif processing_type == 'no_processing':
        output = inputs
    else:
        raise ValueError('Processing type name in encoder of the DP-Transformer model is not supported.')

    for i in range(len(neuron_list)):
        output = tf.keras.layers.Conv2D(neuron_list[i], kernel_size=kernel_size, strides=stride, padding='same')(output)
        # After the index setting, the upsampling layer will be used
        if i == upsampling_layer_idx:
            scale = math.log2(input_size_info[1])
            for i in range(int(scale)):
                if upsampling_type == 'transposed':
                    output = tf.keras.layers.Conv2DTranspose(filters=neuron_list[i], padding='valid', strides=(2,2), kernel_size=kernel_size)(output)
                elif upsampling_type == 'shuffle':
                    output = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2, data_format='NHWC'))(output)
                else:
                    raise ValueError('Upsampling type name in CNN_simple model is not supported.')
    output = tf.keras.layers.Conv2D(last_unit, kernel_size=kernel_size, strides=stride, padding='same')(output)

    if processing_type == 'padding':
        output = output[:, :-(2 * input_size_info[1] - 1), :, :]
    elif processing_type == 'conv':
        output = tf.keras.layers.Conv2DTranspose(2, kernel_size=(2, 1), strides=(1, 1), padding='valid')(output)
    elif processing_type == 'no_processing':
        output = output[:, :-(input_size_info[1] - 1), :, :]
    else:
        raise ValueError('Processing method name in DP-Transformer model is not supported.')

    return tf.keras.Model(inputs=inputs, outputs=output, name='CNN_simple')


@gin.configurable
def UNet_simple(input_size_info, kernel_size, downsample_neuron_list, upsample_neuron_list, processing_method, stride=2):
    '''
    Define a simple UNet model
    :param input_size_info: [%batch_size, %sampling_rate, %frame_length, %num_samples, %num_chirps]
    :return:
    '''
    # set the input
    processing_type, upsampling_type, separation_type, log_type, abs_normalization_type, angle_normalization_type = processing_method.split('&')

    if separation_type == 'ap/ph' or separation_type == 're/im':
        input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2] * 2)
        last_unit = 2
    elif separation_type == 'ap':
        input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2])
        last_unit = 1
    else:
        raise ValueError('The separation type is not supported in unet_concat function.')
    inputs = tf.keras.Input(shape=input_size, dtype=tf.float32)

    # Encoder
    output = tf.keras.layers.Conv2D(8, kernel_size=(2, 1), strides=(1,1), padding='valid')(inputs)
    for i in range(len(downsample_neuron_list)):
        output = tf.keras.layers.Conv2D(downsample_neuron_list[i], kernel_size=kernel_size, strides=(1,1), padding='same')(output)
        output = tf.keras.layers.Conv2D(downsample_neuron_list[i], kernel_size=kernel_size, strides=stride, padding='same')(output)

    # Decoder
    output = tf.keras.layers.Conv2DTranspose(upsample_neuron_list[0], kernel_size=kernel_size, strides=(1, 1), padding='same')(output)
    for i in range(len(upsample_neuron_list)):
        output = tf.keras.layers.Conv2DTranspose(upsample_neuron_list[i], kernel_size=kernel_size, strides=(1, 1), padding='same')(output)
        output = tf.keras.layers.Conv2DTranspose(upsample_neuron_list[i], kernel_size=kernel_size, strides=stride, padding='same')(output)

    output = tf.keras.layers.Conv2DTranspose(last_unit, kernel_size=(2, 1), strides=1, padding='valid')(output)

    return tf.keras.Model(inputs=inputs, outputs=output, name='UNet_simple')


@gin.configurable
def UNet_concat(input_size_info, kernel_size, stride, processing_method, large_model=False):
    '''
    Define a simple UNet model
    :param input_size_info: [%batch_size, %sampling_rate, %frame_length, %num_samples, %num_chirps]
    :return:
    '''
    processing_type, upsampling_type, separation_type, log_type, abs_normalization_type, angle_normalization_type = processing_method.split('&')

    if separation_type == 'ap/ph' or separation_type == 're/im':
        input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2]*2)
        nr_signals = 2
    elif separation_type == 'ap':
        input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2])
        nr_signals = 1
    else:
        raise ValueError('The separation type is not supported in unet_concat function.')
    inputs = tf.keras.Input(shape=input_size, dtype=tf.float32)

    # input_size = (input_size_info[3] // input_size_info[1] // 2 + 1, input_size_info[4] // input_size_info[1], input_size_info[2])
    # inputs = tf.keras.Input(shape=input_size, dtype=tf.complex64)
    # if separation_type == 're/im':
    #     output_0 = re_im_separation(inputs)
    # elif separation_type == 'ap/ph' and log_type == 'no_log':
    #     output_0, mean, max = amplitude_phase_separation(log_type='no_log')(inputs)
    # elif separation_type == 'ap/ph' and log_type == 'with_log':
    #     output_0, mean, max = amplitude_phase_separation(log_type='with_log')(inputs)

    # Encoder
    output_0 = tf.keras.layers.Conv2D(16, kernel_size=(2, 1), strides=(1,1), padding='valid')(inputs)
    output_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(output_0)
    output_1 = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, strides=(1, 1), padding='same')(output_1)
    output_1 = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, strides=(1, 1), padding='same')(output_1)
    output_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(output_1)

    if large_model:
        output_2 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, strides=(1, 1), padding='same')(output_2)
        output_2 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, strides=(1, 1), padding='same')(output_2)
        output_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(output_2)
        output_3 = tf.keras.layers.Conv2D(128, kernel_size=kernel_size, strides=(1, 1), padding='same')(output_3)
        output_3 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, strides=(1, 1), padding='same')(output_3)

        # Decoder
        output = tf.keras.layers.Conv2DTranspose(64, kernel_size=kernel_size, strides=stride, padding='same')(output_3)
        output = tf.concat([output, output_2], 3)
        output = tf.keras.layers.Conv2DTranspose(64, kernel_size=kernel_size, strides=(1, 1), padding='same')(output)
        output = tf.keras.layers.Conv2DTranspose(32, kernel_size=kernel_size, strides=(1, 1), padding='same')(output)

        output = tf.keras.layers.Conv2DTranspose(32, kernel_size=kernel_size, strides=stride, padding='same')(output)
    else:
        output = tf.keras.layers.Conv2DTranspose(32, kernel_size=kernel_size, strides=stride, padding='same')(output_2)
    output = tf.concat([output, output_1], 3)
    output = tf.keras.layers.Conv2DTranspose(32, kernel_size=kernel_size, strides=(1, 1), padding='same')(output)
    output = tf.keras.layers.Conv2DTranspose(16, kernel_size=kernel_size, strides=(1, 1), padding='same')(output)
    output = tf.keras.layers.Conv2DTranspose(16, kernel_size=kernel_size, strides=stride, padding='same')(output)
    output = tf.concat([output, output_0], 3)
    output = tf.keras.layers.Conv2DTranspose(16, kernel_size=kernel_size, strides=(1, 1), padding='same')(output)
    # output = tf.keras.layers.Conv2DTranspose(8, kernel_size=kernel_size, strides=stride, padding='same')(output)

    scale = math.log2(input_size_info[1])
    for i in range(int(scale)):
        output = tf.keras.layers.Conv2DTranspose(8, kernel_size=kernel_size, strides=stride, padding='same')(output)
    # output = tf.keras.layers.Conv2DTranspose(2, kernel_size=(2, 1), strides=(1, 1), padding='valid')(output)
    output = tf.keras.layers.Conv2DTranspose(nr_signals, kernel_size=(2, 1), strides=(1, 1), padding='valid')(output)

    return tf.keras.Model(inputs=inputs, outputs=output, name='UNet_concat')
