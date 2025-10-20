#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:52:08 2024

@author: m145
"""

import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def open_radar(x, y, idx):

    folder_path = '../../../data/data_static/'+str(x)+'_'+str(y)
    with open(folder_path +'/'+str(idx)+'.pickle', 'rb') as f:
        radar_cube = pickle.load(f)
        
    return radar_cube

cube = open_radar(2.0, 2.0, 1)

# we should integrate the following code directly when recording and saving data.
# after this processing of the raw radar data, we save the result (cube) as ground truth.

# de-biasing each chirp
avgs = np.average(cube, 1)[:, None]
cube = cube - avgs

# apply a high-pass filter to fast-time samples (don't know if we actually need this).
# this gets rid of the strong reflections around r=0.
# can be tuned with filter length (first arg) and cut-off frequency (second arg).
# just comment this out or change the parameter values and run again to see the effect.
sos = signal.butter(2, 2e4, 'hp', fs=1.28e6, output='sos')
cube = signal.sosfilt(sos, cube, axis=-1)

# save now after de-biasing and hp-filtering (check with some data later if you actually need the high-pass
# filter. I think de-biasing should actually be enough if the reflections around r=0 are not always
# very dominant.)

################################################################################

# tf rD processing
cube = tf.convert_to_tensor(cube, dtype=tf.float32)

# Tensorflow only has "hann" and "hamming" windows built in. scipy has some more.
# I think "hamming" is actually the better window (I also switched to "hamming" in the numpy code).
# This will give a sharper rD map (hamming window has smaller main lobe in frequency domain than blackmanharris)
range_window = tf.signal.hamming_window(cube.shape[1])
# range_window = signal.windows.blackmanharris(cube.shape[1]).astype(np.float32)
# range_window = tf.convert_to_tensor(range_window)
doppler_window = tf.signal.hamming_window(cube.shape[0])
# doppler_window = signal.windows.blackmanharris(cube.shape[0]).astype(np.float32)
# doppler_window = tf.convert_to_tensor(doppler_window, dtype=tf.complex64)

# windowing for range
cube_windowed = cube * range_window

# range fft
rfft = tf.signal.rfft(cube_windowed)

# windowing for Doppler
rfft = tf.transpose(rfft)
# plt.plot(tf.math.real(rfft[1]), color='red')
# plt.plot(tf.math.imag(rfft[1]), color='orangered')

re_rfft = tf.math.real(rfft) * doppler_window
im_rfft = tf.math.imag(rfft) * doppler_window
rfft = tf.complex(re_rfft, im_rfft)

# Doppler fft and centering of Doppler (0 Doppler in middle)
# rD is the same as before (complex rD map ~ voltage) but without all the
# scaling. This goes to the first trainable layer (or some preprocessing before that).
rD = tf.signal.fft(rfft)
rD = tf.signal.fftshift(rD, 1)

# plot rD result
plt.figure()
plt.pcolormesh(10*tf.experimental.numpy.log10(tf.math.abs(rD)**2).numpy())

# iFFT processing (for network output rD)

# reverse Doppler frequency shifting
rD = np.fft.ifftshift(rD, 1)

# Doppler iFFT
rfft = tf.signal.ifft(rD)

# reverse Doppler windowing
re_rfft = tf.math.real(rfft) / doppler_window
im_rfft = tf.math.imag(rfft) / doppler_window

rfft = tf.complex(re_rfft, im_rfft)

# plt.plot(tf.math.real(rfft[1]), color='black')
# plt.plot(tf.math.imag(rfft[1]), color='grey')

# range iFFT
rfft = tf.transpose(rfft)
cube2 = tf.signal.irfft(rfft)

# reverse range windowing
cube2 = cube2 / range_window


print("Approx. invertible processing", np.allclose(cube, cube2.numpy(), atol=1e-5))