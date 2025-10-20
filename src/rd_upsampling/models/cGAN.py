import keras
import gin
import tensorflow as tf

from models.swinir_tensorflow import *
from models.layers import re_im_separation, ReImSeparation
from models.dp_transformer_stft_exponential import DPTransformerSTFT
from models.swinir_tensorflow import SwinIR
from models.architectures import UNet_concat


class Discriminator_pix2pix(keras.Model):
    '''The discriminator with the low-resolution data as the condition'''
    def __init__(self):
        super().__init__()

        self.inp_downsample1 = self.build_downsample(16, 4, 2, True)
        # self.inp_downsample2 = self.build_downsample(32, 4, 2)

        self.tar_downsample1 = self.build_downsample(8, 4, 2, True)
        self.tar_downsample2 = self.build_downsample(16, 4, 2)
        # self.tar_downsample3 = self.build_downsample(16, 4, 2)

        self.out_downsample1 = self.build_downsample(32, 4, 2)
        self.out_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def build_downsample(self, filters, size, strides, first_downsample=False):
        result = tf.keras.Sequential()
        if first_downsample:
            result.add(tf.keras.layers.Conv2D(2, kernel_size=(2, 1), strides=1, padding='valid'))
        result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def call(self, inp, tar):
        inp = abs(inp) ** 2
        inp_down = self.inp_downsample1(inp)  # (batch_size, 129, 32, 1) --> (batch_size, 64, 16, 16)
        # inp_down = self.inp_downsample2(inp_down)  # --> (batch_size, 32, 8, 32)

        tar = abs(tar) ** 2
        tar_down = self.tar_downsample1(tar)  # (batch_size, 257, 64, 1) --> (batch_size, 128, 32, 8)
        tar_down = self.tar_downsample2(tar_down)  # --> (batch_size, 64, 16, 16)
        # tar_down = self.tar_downsample3(tar_down)  # --> (batch_size, 32, 8, 32)

        down = tf.keras.layers.concatenate([inp_down, tar_down], axis=-1)  # (batch_size, 32, 8, 32*2)

        out = self.out_downsample1(down)  # (batch_size, 16, 4, 64)
        out = self.out_dense(out)  # (batch_size, 16, 4, 1)

        return out


class Discriminator(keras.Model):
    '''The discriminator without the low-resolution data as the condition'''
    def __init__(self, scale='small'):
        super().__init__()

        self.tar_downsample1 = self.build_downsample(8, 4, 2, True)
        self.tar_downsample2 = self.build_downsample(16, 4, 2)
        self.tar_downsample3 = self.build_downsample(32, 4, 2)
        if scale == 'large':
            self.tar_downsample4 = self.build_downsample(64, 4, 2)
            self.tar_downsample5 = self.build_downsample(128, 4, 2)
            self.tar_downsample6 = self.build_downsample(256, 4, 2)
            # self.tar_downsample7 = self.build_downsample(512, 4, 1)
        else:
            pass

        self.out_dense = tf.keras.layers.Dense(1, activation='sigmoid')
        self.scale = scale

    def build_downsample(self, filters, size, strides, first_downsample=False):
        result = tf.keras.Sequential()
        if first_downsample:
            result.add(tf.keras.layers.Conv2D(2, kernel_size=(2, 1), strides=1, padding='valid'))
        result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.PReLU())
        return result

    def call(self, tar):

        # tar = abs(tar) ** 2
        tar_down = self.tar_downsample1(tar)  # (batch_size, 257, 64, 1) --> (batch_size, 128, 32, 8)
        tar_down = self.tar_downsample2(tar_down)  # --> (batch_size, 64, 16, 16)
        tar_down = self.tar_downsample3(tar_down)  # --> (batch_size, 32, 8, 32)
        if self.scale == 'large':
            tar_down = self.tar_downsample4(tar_down)
            tar_down = self.tar_downsample5(tar_down)
            tar_down = self.tar_downsample6(tar_down)
            # tar_down = self.tar_downsample7(tar_down)
        else:
            pass

        down = tf.reshape(tar_down, (tar_down.shape[0], -1))
        out = self.out_dense(down)  # (batch_size, 1)

        return out


@gin.configurable
class ConditionalGAN(keras.Model):
    def __init__(self, discriminator_name, generator_name, discriminator_scale, input_size_info):
        super().__init__()
        # Load the generator model with the model name
        if generator_name == 'SwinIR':
            self.generator = SwinIR()
        elif generator_name == 'UNet_concat':
            self.generator = UNet_concat()
        elif generator_name == 'DP-Transformer':
            self.generator = DPTransformerSTFT()
        else:
            raise ValueError('Generator name in cGAN model is not supported.')

        # Load the discriminator model with the model name
        if discriminator_name == 'discriminator':
            self.discriminator = Discriminator(scale=discriminator_scale)
            # self.discriminator = Discriminator_pix2pix()

    def call(self, inputs, type, targets=None, **kwargs):
        if type == 'generator':
            return self.generator(inputs, training=kwargs.get('training', False))
        elif type == 'discriminator':
            return self.discriminator(inputs, training=kwargs.get('training', False))
        else:
            raise ValueError('Type name in cGAN model is not supported.')
    # def call(self, inputs, type):
    #     if type == 'generator':
    #         return self.generator(inputs)
    #     elif type == 'discriminator':
    #         return self.discriminator(inputs)
    #     else:
    #         raise ValueError('Type name in cGAN model is not supported.')
