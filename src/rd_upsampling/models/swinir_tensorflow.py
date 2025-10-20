import gin
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, Dropout, LayerNormalization
from tensorflow.keras import Model, Sequential
import math
# import keras
from tensorflow.keras import layers
import numpy as np

from models.layers import re_im_separation, ReImSeparation, amplitude_phase_separation, amplitude_phase_postprocessing
from models.dp_transformer_stft_exponential import DPTransformerBase


def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size[0]
    patch_num_x = width // window_size[1]
    x = tf.reshape(x, (-1, patch_num_y, window_size[0], patch_num_x, window_size[1], channels))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, (-1, window_size[0], window_size[1], channels))
    return windows


def window_reverse(windows, window_size, num_patch, height, width, channels):
    window_num_y = num_patch[0] // window_size[0]
    window_num_x = num_patch[1] // window_size[1]
    x = tf.reshape(
        windows,
        (
            -1,
            window_num_y,
            window_num_x,
            window_size[0],
            window_size[1],
            channels,
        ),
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (-1, num_patch[0], num_patch[1], channels))
    return x


class PatchExtract(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def call(self, x):
        batch_size, height, width, channels = x.shape
        x = tf.reshape(x, (batch_size, height * width, channels))
        x = tf.reshape(x, (batch_size, height // self.patch_size_x, self.patch_size_x, width // self.patch_size_y, self.patch_size_y, channels))
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, (batch_size, (height // self.patch_size_x) * (width // self.patch_size_y), self.patch_size_x * self.patch_size_y * channels))

        return x


class PatchReconstruct(tf.keras.layers.Layer):
    def __init__(self, patch_size, original_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]
        self.original_height = original_size[1]
        self.original_width = original_size[2]

    def __call__(self, x):
        batch_size, num_patches, patch_dim = x.shape
        x = tf.reshape(x, (batch_size, self.original_height//self.patch_size_x,  self.original_width//self.patch_size_y, patch_dim))
        grid_height = self.original_height // self.patch_size_y
        grid_width = self.original_width // self.patch_size_x
        x = tf.reshape(x, (batch_size, grid_height, grid_width, self.patch_size_y, self.patch_size_x, -1))
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, (batch_size, self.original_height, self.original_width, -1))

        return x



class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.experimental.numpy.arange(start=0, stop=self.num_patch)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.shape
        x = tf.reshape(x, (-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concatenate((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, (-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)


class WindowAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=relative_position_index,
            dtype=tf.int32,
            trainable=False,
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        # x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x, (-1, size, 1, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, (2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[0], x_qkv[0]
        q = q * self.scale
        k = tf.transpose(k, (0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, (-1,))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            relative_position_index_flat,
            axis=0,
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            (num_window_elements, num_window_elements, -1),
        )
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0),
                "float32",
            )
            attn = tf.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = tf.reshape(attn, (-1, self.num_heads, size, size))
            attn = tf.keras.activations.softmax(attn, axis=-1)
        else:
            attn = tf.keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, (0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, (-1, size, channels))
        # x_qkv = self.proj(x_qkv)
        # x_qkv = self.dropout(x_qkv)
        return x_qkv


class SwinTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        x_size,
        num_heads,
        window_size=[5,2],
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0
    ):
        super().__init__()

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.x_size = x_size  # size of original input
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes
        self.attn_mask = None

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(dim),
                layers.Activation(tf.keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(num_mlp),
                layers.Dropout(dropout_rate),
            ]
        )

        if self.num_patch[0] < self.window_size[0] or self.num_patch[1] < self.window_size[1]:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    # def build(self, input_shape):
    #     if self.shift_size == 0:
    #
    #     else:
    #         height, width = self.num_patch
    #         h_slices = (
    #             slice(0, -self.window_size),
    #             slice(-self.window_size, -self.shift_size),
    #             slice(-self.shift_size, None),
    #         )
    #         w_slices = (
    #             slice(0, -self.window_size),
    #             slice(-self.window_size, -self.shift_size),
    #             slice(-self.shift_size, None),
    #         )
    #         mask_array = np.zeros((1, height, width, 1))
    #         count = 0
    #         for h in h_slices:
    #             for w in w_slices:
    #                 mask_array[:, h, w, :] = count
    #                 count += 1
    #         mask_array = tf.convert_to_tensor(mask_array)
    #
    #         # mask array to windows
    #         mask_windows = window_partition(mask_array, self.window_size)
    #         mask_windows = tf.reshape(
    #             mask_windows, [-1, self.window_size * self.window_size]
    #         )
    #         attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
    #             mask_windows, axis=2
    #         )
    #         attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
    #         attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
    #         self.attn_mask = tf.Variable(
    #             initializer=attn_mask,
    #             shape=attn_mask.shape,
    #             dtype=attn_mask.dtype,
    #             trainable=False,
    #         )

    def call(self, x, training=True):
        height, width = self.x_size
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, self.num_patch[0], self.num_patch[1], channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size[0] * self.window_size[1], channels))

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size[0], self.window_size[1], channels))
        shifted_x = window_reverse(attn_windows, self.window_size, self.num_patch, height, width, channels)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, self.num_patch[0] * self.num_patch[1], channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x


class BasicLayer(tf.keras.layers.Layer):
    """A basic Swin Transformer layer for one stage."""

    def __init__(self, dim, depth, num_layer, num_patch_x, num_patch_y, patch_size, inputs_shape, num_mlp=256,
                 qkv_bias=True, dropout_rate=0.03, num_heads=2, d_model=128,
                 dff=128, maximum_position_encoding=10000, re_im_split = 0,
                 transformer_type='DP', window_size = 7):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.transformer_type = transformer_type
        self.inputs_shape = inputs_shape
        self.num_patch_x = num_patch_x
        self.num_patch_y = num_patch_y

        # Build blocks
        self.blocks = tf.keras.models.Sequential()  # the loop of Swin or DP Transformer blocks
        if self.transformer_type == 'Swin':
            self.blocks.add(PatchExtract(patch_size))
            for i in range(depth):
                self.blocks.add(SwinTransformer(
                    dim=dim,
                    num_patch=(num_patch_x, num_patch_y),
                    x_size= (inputs_shape[1], inputs_shape[2]),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=1,
                    num_mlp=num_mlp,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate,
                ))
            self.blocks.add(PatchReconstruct(patch_size, inputs_shape))
        elif self.transformer_type == 'DP':
            for i in range(depth):
                self.blocks.add(DPTransformerBase(num_layers=num_layer,
                                      d_model=d_model,
                                      num_heads=num_heads,
                                      dff=dff,
                                      rate=.1,
                                      maximum_position_encoding=maximum_position_encoding,
                                      re_im_split=re_im_split))


    def call(self, x):
        x = self.blocks(x)

        return x


class RSTB(tf.keras.layers.Layer):
    """Residual Swin Transformer Block (RSTB)."""

    def __init__(self, dim, depth, num_layer, num_patch_x, num_patch_y, patch_size, swin_inputs_shape, num_mlp,
                 qkv_bias, dropout_rate, num_heads, d_model, dff,
                 maximum_position_encoding, re_im_split, transformer_type, window_size, kernel_size):

        super(RSTB, self).__init__()

        # Residual group using BasicLayer (assumes BasicLayer is implemented in TensorFlow)
        self.residual_group = BasicLayer(dim, depth, num_layer, num_patch_x, num_patch_y, patch_size, swin_inputs_shape, num_mlp = num_mlp,
                                         qkv_bias = qkv_bias, dropout_rate = dropout_rate, num_heads=num_heads,
                                         d_model=d_model, dff=dff,
                                         maximum_position_encoding=maximum_position_encoding,
                                         re_im_split = re_im_split, transformer_type=transformer_type, window_size = window_size)

        # Residual connection
        self.conv = tf.keras.layers.Conv2D(dim, kernel_size=kernel_size, strides=1, padding='same')

    def call(self, x):
        # Forward pass
        residual = x
        x = self.residual_group(x)
        # x = self.conv(x)
        return x + residual


@gin.configurable
class SwinIR(Model):
    def __init__(self, transformer_type, processing_method, input_size_info, embed_dim_swin, embed_dim_df, dp_depths, dp_num_layer, swin_depths, swin_num_layer, patch_size=(2,2), num_mlp=256,
                 qkv_bias=True, dropout_rate=0.03, num_heads=2, d_model=96, dff=96,
                 maximum_position_encoding=10000, re_im_split=0, kernel_size=3,
                 upsampling_ratio=2, window_size=7):
        super().__init__()

        self.processing_type, self.upsampling_type, self.separation_type, self.log_type, self.abs_normalization_type, self.angle_normalization_type = processing_method.split('&')
        self.upsampling_ratio = upsampling_ratio
        self.input_size_info = input_size_info
        if transformer_type == 'DP':
            self.num_layers = len(dp_depths)
            depths = dp_depths
            self.num_layer = dp_num_layer
            embed_dim = embed_dim_df
        elif transformer_type == 'Swin':
            self.num_layers = len(swin_depths)
            depths = swin_depths
            self.num_layer = swin_num_layer
            embed_dim = embed_dim_swin
        else:
            raise ValueError('Transformer type in SwinIR model is not supported.')
        num_feat = 96
        if self.separation_type == 'ap/ph' or self.separation_type == 're/im':
            num_out_ch = 2
        elif self.separation_type == 'ap':
            num_out_ch = 1
        else:
            raise ValueError('Separation type in SwinIR model is not supported.')

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################

        self.conv_first = Conv2D(embed_dim, kernel_size=kernel_size, strides=1, padding='same')
        self.separable_conv = tf.keras.layers.SeparableConv2D(8, kernel_size=(2, 1), strides=(1, 1), padding='valid')

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.conv_after_body = tf.keras.layers.Conv2D(embed_dim, kernel_size=kernel_size, strides=1, padding='same')
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(2, kernel_size=(2, 1), strides=(1, 1), padding='valid')

        self.pos_drop = tf.keras.layers.Dropout(rate=dropout_rate)

        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

        # num_patch_x = self.input_size_info[3]//self.input_size_info[1]//patch_size[0]
        # num_patch_y = self.input_size_info[4]//self.input_size_info[1]//patch_size[1]

        if self.processing_type == 'padding':
            num_patch_x = (self.input_size_info[3]//self.input_size_info[1]//2+2) // patch_size[0]
            num_patch_y = self.input_size_info[4] // self.input_size_info[1] // patch_size[1]
            swin_inputs_shape = (self.input_size_info[0], self.input_size_info[3]//self.input_size_info[1]//2+2, self.input_size_info[4] // self.input_size_info[1], embed_dim)
        elif self.processing_type == 'conv':
            num_patch_x = (self.input_size_info[3] // self.input_size_info[1] // 2) // patch_size[0]
            num_patch_y = self.input_size_info[4] // self.input_size_info[1] // patch_size[1]
            swin_inputs_shape = (self.input_size_info[0], self.input_size_info[3]//self.input_size_info[1]//2, self.input_size_info[4] // self.input_size_info[1], embed_dim)
        elif self.processing_type == 'no_processing':
            num_patch_x = None
            num_patch_y = None
            swin_inputs_shape = (self.input_size_info[0], self.input_size_info[3] // self.input_size_info[1] // 2 + 1, self.input_size_info[4] // self.input_size_info[1], embed_dim)
        else:
            raise ValueError('Processing type in DP-Transformer model is not supported.')

        self.RSTB_layers = tf.keras.models.Sequential()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim, depth=depths[i_layer], num_layer=self.num_layer, num_patch_x=num_patch_x, num_patch_y=num_patch_y,
                         patch_size=patch_size, swin_inputs_shape=swin_inputs_shape, num_mlp=num_mlp, qkv_bias=qkv_bias,
                         dropout_rate=dropout_rate, num_heads=num_heads, d_model=d_model, dff=dff,
                         maximum_position_encoding=maximum_position_encoding,
                         re_im_split=re_im_split, transformer_type=transformer_type, window_size=window_size, kernel_size=kernel_size)
            self.RSTB_layers.add(layer)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        # self.conv_before_upsample = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(filters=num_feat, kernel_size=3, strides=1, padding='same'),
        #     tf.keras.layers.LeakyReLU()
        # ])

        self.upsample = tf.keras.Sequential()
        scale = math.log2(upsampling_ratio)
        for _ in range(int(scale)):
            if self.upsampling_type == 'transposed':
                self.upsample.add(tf.keras.layers.Conv2DTranspose(filters=embed_dim, kernel_size=4, strides=2, activation=None, padding='same'))
                self.upsample.add(tf.keras.layers.LayerNormalization())
                self.upsample.add(tf.keras.layers.ReLU())
            # self.upsample.add(tf.keras.layers.Conv2D(filters=num_feat, kernel_size=3, strides=1, padding='same'))
            elif self.upsampling_type == 'shuffle':
                self.upsample.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2, data_format='NHWC')))

        self.conv_last = tf.keras.layers.Conv2D(filters=num_out_ch, kernel_size=kernel_size, strides=1, padding='same')

    def forward_features(self, x):

        x = self.pos_drop(x)

        x = self.RSTB_layers(x) # the loop of RSTB blocks

        x = self.norm(x)  # B L C

        return x


    def call(self, x):

        if self.processing_type == 'padding':
            paddings = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]])
            x = tf.pad(x, paddings, mode='CONSTANT', constant_values=0)
        elif self.processing_type == 'conv':
            x = self.separable_conv(x)

        # x = self.conv_first(x)
        # x = self.conv_after_body(self.forward_features(x)) + x
        # x = self.conv_before_upsample(x)
        # x = self.conv_last(self.upsample(x))

        x = self.conv_first(x)
        x = self.forward_features(x) + x
        # x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        if self.processing_type == 'padding':
            x = x[:, :-(2*self.upsampling_ratio-1), :, :]
        elif self.processing_type == 'conv':
            x = self.conv_transpose(x)
        elif self.processing_type == 'no_processing':
            x = x[:, :-(self.upsampling_ratio - 1), :, :]
        else:
            raise ValueError('Processing method name in DP-Transformer model is not supported.')

        return x

    def model(self):
        x = tf.keras.Input(shape=(self.input_size_info[3]//self.input_size_info[1]//2+1, self.input_size_info[4]//self.input_size_info[1], self.input_size_info[2]))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
