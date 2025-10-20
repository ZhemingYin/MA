#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Path Transformer Architecture.
We "simply" replace the DP-RNNs of the Complex DP-RNN Network with Transformer
Encoders, similar to this:
    https://arxiv.org/abs/2010.13154

This is the version that uses complex exponential inputs. The input amplitudes
are log-transformed and the output amplitudes exp-transformed later
to deal with amplitude differences of real radar scenarios.
--> log-spectrogram and phase (-pi, pi) as input.
we use only log10(x) and 10^x instead of 10*log10(x) and 10^(x/10) to reduce
the predicted value range of the log-spec and adapt it more to the phase value range
since we assume this should make training the nets easier and lead to better results.
"""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import gin
import math
from tensorflow.keras import mixed_precision
from models.layers import re_im_separation, amplitude_phase_separation, amplitude_phase_postprocessing


# The transformer functions here are copied from the transformer tutorial found at:
# https://www.tensorflow.org/text/tutorials/transformer with some minor adjustments.
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


# Convolutional Encoding of the complex STFT inputs
class Encoder(tf.keras.layers.Layer):

    def __init__(self, kernel_size=3, filters=64, strides=[1, 1], processing_type='padding'):
        super(Encoder, self).__init__()
        self.processing_type = processing_type
        self.range_preprocessing = tf.keras.layers.SeparableConv2D(8, kernel_size=(2, 1), strides=(1, 1), padding='valid')

        # input convolution
        self.input_conv1 = tf.keras.layers.SeparableConv2D(filters=filters,
                                                           kernel_size=kernel_size,
                                                           strides=[1, 1],
                                                           data_format='channels_last',
                                                           padding='same',
                                                           activation=None)
        self.input_conv_ln1 = tf.keras.layers.LayerNormalization()
        # down convolution for the bigger transformer architectures with compressed inputs
        self.down_conv = tf.keras.layers.SeparableConv2D(filters=filters,
                                                         kernel_size=kernel_size,
                                                         strides=strides,
                                                         data_format='channels_last',
                                                         padding='same',
                                                         activation=None)
        self.down_conv_ln = tf.keras.layers.LayerNormalization()
        self.down_conv_act = tf.keras.layers.ReLU()

    def call(self, inputs):

        if self.processing_type == 'padding':
            paddings = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]])
            inputs_exponential = tf.pad(inputs, paddings, mode='CONSTANT', constant_values=0)
        elif self.processing_type == 'conv':
            inputs_exponential = self.range_preprocessing(inputs)
        elif self.processing_type == 'no_processing':
            inputs_exponential = inputs
        else:
            raise ValueError('Processing type name in encoder of the DP-Transformer model is not supported')

        # input conv
        input_conv1 = self.input_conv1(inputs_exponential)
        input_conv1 = self.input_conv_ln1(input_conv1)
        # downsampling conv (with strides > 1)
        enc = self.down_conv(input_conv1)
        enc = self.down_conv_ln(enc)
        enc = self.down_conv_act(enc)
        return enc


# Linear Decoding the 3-D embeddings from separator
class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_filters, nr_signals, kernel_size, upsampling_ratio, strides=[1, 1], upsampling_method='shuffle'):
        super(Decoder, self).__init__()
        self.upsampling_ratio = upsampling_ratio
        self.upsampling_method = upsampling_method

        # decoder output convolution
        self.dec = keras.layers.Conv2D(filters=nr_signals,
                                       kernel_size=kernel_size,
                                       strides=[1, 1], padding='same',
                                       activation=None)

        self.upsample = tf.keras.Sequential()
        self.scale = math.log2(upsampling_ratio)
        for _ in range(int(self.scale)):
            if self.upsampling_method == 'transposed':
                self.upsample.add(keras.layers.Conv2DTranspose(filters=dec_filters, kernel_size=4, strides=2, activation=None, padding='same'))
                self.upsample.add(keras.layers.LayerNormalization())
                self.upsample.add(keras.layers.ReLU())
            # self.upsample.add(tf.keras.layers.Conv2D(filters=num_feat, kernel_size=3, strides=1, padding='same'))
            elif self.upsampling_method == 'shuffle':
                self.upsample.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2, data_format='NHWC')))

    def call(self, inputs):

        up = self.upsample(inputs)
        out = self.dec(up)

        return out


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        # q = self.wq(q)  # (batch_size, seq_len, d_model)
        # k = self.wk(k)  # (batch_size, seq_len, d_model)
        # v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# Dual Path Transformer
class TransformerSeparator(tf.keras.layers.Layer):
    def __init__(self, num_dp_layers, num_layers,
                 filters, d_model, dff, num_heads,
                 maximum_position_encoding=10000,
                 re_im_split=False):
        super(TransformerSeparator, self).__init__()

        self.input_conv = keras.models.Sequential(
            [keras.layers.Conv2D(filters,
                                 kernel_size=1, strides=1, activation=None, padding='same'),
             keras.layers.LayerNormalization()])

        self.model = keras.models.Sequential()
        # stack of dual-path transformer blocks
        for i in range(num_dp_layers):
            self.model.add(DPTransformerBase(num_layers=num_layers,
                                             d_model=d_model,
                                             num_heads=num_heads,
                                             dff=dff,
                                             rate=.1,
                                             maximum_position_encoding= \
                                                 maximum_position_encoding,
                                             re_im_split=re_im_split))

        self.left_path = keras.models.Sequential(
            [keras.layers.Conv2D(filters, 1, 1, activation=None, padding='same'),
             keras.layers.Activation('tanh')])

        # Right Conv1D with Sigmoid
        self.right_path = keras.models.Sequential(
            [keras.layers.Conv2D(filters, 1, 1, activation=None, padding='same'),
             keras.layers.Activation('sigmoid')])

        # Tail of Separator Module
        self.tail = keras.models.Sequential(
            [keras.layers.Conv2D(filters, 1, 1, activation=None, padding='same'),
             keras.layers.LayerNormalization()])

        # self.conv = keras.layers.SeparableConv2D(filters = filters,
        #                                           kernel_size = 1, strides = 1,
        #                                           padding='same')
        # self.conv_act = keras.layers.ReLU()

    def call(self, inputs):
        input_conv = self.input_conv(inputs)
        # dp_trans_out = self.model(inputs)
        dp_trans_out = self.model(input_conv)

        gated_dprnn_out = tf.math.multiply(self.left_path(dp_trans_out),
                                           self.right_path(dp_trans_out))

        separator_masks = self.tail(gated_dprnn_out)
        # multiplication of input and masks
        masked_output = tf.math.multiply(separator_masks, inputs)
        out = masked_output
        # out = dp_trans_out
        return out


# standard transformer encoding layer with ln, mha and ffn
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        # out1 = x + attn_output
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        # out2 = out1 + ffn_output
        return out2


# Here we first apply the intra-transformer, then the inter-transformer.
# Same as the DP-RNN only with Transformers instead of bidirectional LSTMs/GRUs.
# We apply pos encoding before each intra and inter block as in the mila paper
# since we need to inject positional information for both dimensions.
class DPTransformerBase(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding=10000, rate=.1,
                 re_im_split=False):
        super(DPTransformerBase, self).__init__()
        self.d_model = d_model
        self.re_im_split = re_im_split
        self.num_layers = num_layers
        if re_im_split:
            self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                    d_model // 2)
        else:
            self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                    d_model)
        # intra transformer layers
        self.enc_intra_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate)
                                 for _ in range(num_layers)]
        # inter transformer layers (here same structure as intra layers for
        # simplicity, might change later)
        self.enc_inter_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate)
                                 for _ in range(num_layers)]
        # self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        B, S, K, N = x.shape
        # input of intra transformer
        intra_in = x
        # reshape for intra block (attention on time-axis)
        intra_out = tf.reshape(tensor=intra_in, shape=(tf.shape(x)[0] * S, K, N))
        # positional encoding if re and im parts are split
        if self.re_im_split:
            intra_out_real = intra_out[:, :, :N // 2]
            intra_out_imag = intra_out[:, :, N // 2:]
            intra_out_real += self.pos_encoding[:, :K, :]
            intra_out_imag += self.pos_encoding[:, :K, :]
            intra_out = tf.concat([intra_out_real, intra_out_imag], axis=-1)
        # standard pos encoding
        else:
            intra_out += self.pos_encoding[:, :K, :]
        # loop over all layers of the intra-transformer
        for i in range(self.num_layers):
            intra_out = self.enc_intra_layers[i](intra_out, training, mask)
        # reshape back for the residual addition around the intra block
        intra_out = tf.reshape(tensor=intra_out, shape=(tf.shape(x)[0], S, K, N))
        # residual addition around the intra block
        intra_out = keras.layers.Add()([intra_in, intra_out])
        # switch S/K dims and flatten seg_size dimension
        # (same procedure as for intra transformer, only now applied to inter dimension)
        inter_in = intra_out
        # reshape for inter block (attention of f-axis)
        inter_out = tf.transpose(inter_in, perm=[0, 2, 1, 3])
        inter_out = tf.reshape(tensor=inter_out, shape=(tf.shape(x)[0] * K, S, N))
        # positional encoding if re and im parts are split
        if self.re_im_split:
            inter_out_real = inter_out[:, :, :N // 2]
            inter_out_imag = inter_out[:, :, N // 2:]
            inter_out_real += self.pos_encoding[:, :S, :]
            inter_out_imag += self.pos_encoding[:, :S, :]
            inter_out = tf.concat([inter_out_real, inter_out_imag], axis=-1)
        # standard pos encoding
        else:
            inter_out += self.pos_encoding[:, :S, :]
        # loop over all layers of the inter-transformer
        for i in range(self.num_layers):
            inter_out = self.enc_inter_layers[i](inter_out, training, mask)
        # reshape back to original
        inter_out = tf.reshape(tensor=inter_out, shape=(tf.shape(x)[0], K, S, N))
        inter_out = tf.transpose(inter_out, perm=[0, 2, 1, 3])
        # residual addition around the inter block
        inter_out = keras.layers.Add()([inter_in, inter_out])
        # # # residual addition around the whole inter- intra stack
        # out = keras.layers.Add()([x, inter_out])
        return inter_out  # (batch_size, input_seq_len, d_model)


@gin.configurable
# The whole separation model consisting of encoder, separator and decoder
class DPTransformerSTFT(tf.keras.Model):
    def __init__(self, processing_method, num_dp_layers, num_layers,
                 d_model, dff, num_heads,
                 enc_kernel_size, dec_kernel_size,
                 enc_filters, dec_filters, sep_filters,
                 strides, upsampling_ratio,
                 input_size_info,
                 maximum_position_encoding=10000,
                 re_im_split=False):
        super(DPTransformerSTFT, self).__init__()
        self.processing_type, self.upsampling_type, self.separation_type, self.log_type, self.abs_normalization_type, self.angle_normalization_type = processing_method.split('&')

        self.enc = Encoder(kernel_size=enc_kernel_size, filters=dff,
                           strides=strides, processing_type=self.processing_type)

        self.sep = TransformerSeparator(num_dp_layers=num_dp_layers,
                                        num_layers=num_layers,
                                        d_model=dff,
                                        dff=dff,
                                        num_heads=num_heads,
                                        filters=dff,
                                        maximum_position_encoding= \
                                            maximum_position_encoding,
                                        re_im_split=re_im_split)

        if self.separation_type == 'ap/ph' or self.separation_type == 're/im':
            nr_signals = 2
        elif self.separation_type == 'ap':
            nr_signals = 1
        else:
            raise ValueError('Separation type in DP-Transformer model is not supported.')

        self.dec = Decoder(nr_signals=nr_signals,
                           dec_filters=dff,
                           kernel_size=dec_kernel_size,
                           upsampling_ratio=upsampling_ratio,
                           strides=strides,
                           upsampling_method=self.upsampling_type)
        self.range_postprocessing = tf.keras.layers.Conv2DTranspose(2, kernel_size=(2, 1), strides=(1, 1), padding='valid')

        # we only use the masking approach (outputs = mixed_input*masks)
        # here, this attr. is needed for saving the model names in a unified
        # format that has the masking in its name (see train function)
        self.masking = True
        self.strides = strides
        self.input_size_info = input_size_info
        self.upsampling_ratio = upsampling_ratio

    def call(self, inputs):
        enc = self.enc(inputs)
        separated = self.sep(enc)
        out = self.dec(separated)

        if self.processing_type == 'padding':
            out = out[:, :-(2 * self.upsampling_ratio - 1), :, :]
        elif self.processing_type == 'conv':
            out = self.range_postprocessing(out)
        elif self.processing_type == 'no_processing':
            out = out[:, :-(self.upsampling_ratio - 1), :, :]
        else:
            raise ValueError('Processing method name in DP-Transformer model is not supported.')

        return out

    def model(self):
        x = tf.keras.Input(shape=(self.input_size_info[3] // self.input_size_info[1] // 2 + 1, self.input_size_info[4] // self.input_size_info[1], self.input_size_info[2]))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))