#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:18:46 2024

Script to setup radar and fetch single frames from one antenna.

The radar python wrapper (ifxradarsdk) is installed on both devices (PC and Raspi).
On the Raspberry this only works with 32 bit Raspberry Pi OS  -->
No Ubuntu since they stopped 32 bit support with version 16 or 18. 

The issue is that pywheels (the library required to build the python wrapper)
doesn't support a 64 bit OS right now: https://blog.piwheels.org/raspberry-pi-os-64-bit-aarch64/'.
There is no workaround (except using something like Ubuntu in Docker or an Ubuntu VM or switching to a 64-bit mini pc).

This is mostly copied from the given example code by Infineon: 
    
~/Documents/radar_sdk_new/examples/py/BGT60TR13C/raw_data.py
~/Documents/radar_sdk_new/examples/py/BGT60TR13C/distance_fft.py

For radar parameter config see Pascals MA, Chenmings PhD or the Radar intro chapter of the
Infineon Radar docs. 
"""

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics, FmcwSequenceChirp
from scipy import signal
import os
from tqdm import tqdm, trange
import pandas as pd
import scipy as sp


def get_radar_cube(device):

    # fetches exactly one data cube for one antenna
    
    frame_contents = device.get_next_frame()
    # raw adc data cube for one frame and one antenna
    data_cube = frame_contents[0][0]

    return data_cube

def fft_spectrum(mat, range_window):
    # see: /home/audiolab/Documents/radar_sdk_new/examples/py/BGT60TR13C/helpers/fft_spectrum.py
    # This does not only use fft1 but also zero padding and scaling
    # Used for range fft
    
    # Calculate fft spectrum
    # mat:          chirp data
    # range_window: window applied on input data before fft

    # received data 'mat' is in matrix form for a single receive antenna
    # each row contains 'num_samples' for a single chirp
    # total number of rows = 'num_chirps'

    # -------------------------------------------------
    # Step 1 - remove DC bias from samples
    # -------------------------------------------------
    [num_chirps, num_samples] = np.shape(mat)

    # helpful in zero padding for high resolution FFT.
    # compute row (chirp) averages
    avgs = np.average(mat, 1).reshape(num_chirps, 1)

    # de-bias values
    mat = mat - avgs
    # -------------------------------------------------
    # Step 2 - Windowing the Data
    # -------------------------------------------------
    mat = np.multiply(mat, range_window)

    # -------------------------------------------------
    # Step 3 - add zero padding here
    # -------------------------------------------------
    # zp1 = np.pad(mat, ((0, 0), (0, num_samples)), 'constant')

    # -------------------------------------------------
    # Step 4 - Compute FFT for distance information
    # -------------------------------------------------
    # range_fft = np.fft.fft(zp1)# / num_samples
    # range_fft = np.fft.fft(mat) #/ num_samples
    range_fft = np.fft.rfft(mat) #/ num_samples
    # ignore the redundant info in negative spectrum
    # compensate energy by doubling magnitude
    range_fft = 2 * range_fft[:, range(int(num_samples//2))]
    
    return range_fft

def init_radar_from_metrics():
    """
    Just some function to set the radar parameters by using the radar metrics.

    The device metrics can be accessed with conversion from sequence: 
        device.metrics_from_sequence(chirp_loop)
    
    We can also convert from metrics to sequence with:
        device.sequence_from_metrics(metrics, chirp_loop)
        
    Or get the chirp params with:
        device.get_acquisition_sequence().chirp
    
    For more details see:
    /home/audiolab/anaconda3/envs/py311/lib/python3.11/site-packages/ifxradarsdk/fmcw/fmcw.py
    
    ToDo: When the parameters are final, throw config parameters into JSON.
    The Infineon radar doc shows how to set this up.

    Parameters
    ----------
    None.

    Returns
    -------
    device : DeviceFmcw object
        The created FMCW device with given.
    chirp_loop : 

    """
    
    # setup 
    device = DeviceFmcw()
    
    metrics = FmcwMetrics(
        range_resolution_m=0.075,
        max_range_m=7.6,
        max_speed_m_s=3,
        speed_resolution_m_s=0.2,
        center_frequency_Hz=61_000_000_000)
    
    # create acquisition sequence based on metrics parameters
    sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
    
    # we fetch radar data at trigger times --> this has no effect
    sequence.loop.repetition_time_s = 1 / 10.  # set frame repetition time

    # convert metrics into chirp loop parameters
    chirp_loop = sequence.loop.sub_sequence.contents
    device.sequence_from_metrics(metrics, chirp_loop)

    # set remaining chirp parameters which are not derived from metrics
    
    # antenna 2 and 3 seem good choices; ant1 looks like it suffers from stronger
    # cross-talk with the tx antenna.
    
    # Physical antenna configuration: See Fig. 30 in radar data sheet
    # rx2    tx1
    # 
    # rx3  rx1
    
    # ant1 = 0, ant2 = 1, ant3 = 2
    rx_antenna_idx = 1
    chirp = chirp_loop.loop.sub_sequence.contents.chirp
    chirp.sample_rate_Hz = 1_280_000 #
    chirp.rx_mask = (1 << rx_antenna_idx) # https://community.infineon.com/t5/Radar-sensor/RX-mask-setting-of-BGT60TR13C/td-p/669486
    chirp.tx_mask = 1
    chirp.tx_power_level = 31
    chirp.if_gain_dB = 33
    chirp.lp_cutoff_Hz = 1280000
    chirp.hp_cutoff_Hz = 80000

    device.set_acquisition_sequence(sequence)
    
    return device, chirp_loop

def init_radar_from_config():
       
    # Same config but with more control over all parameters.  
    # See config_device_from_metrics() 
    # Better use this one if you know what you're doing :)
    device = DeviceFmcw()
    
    rx_antenna_idx = 2
    config = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=1 /10., # fps, not used if data is collected at triggers
    chirp_repetition_time_s=0.000220,    # prt
    num_chirps=64,
    tdm_mimo=False,                         # MIMO disabled
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=59_000_000_000,
        end_frequency_Hz=63_000_000_000,
        sample_rate_Hz=1_280_000*2,         # We need to double the sample rate from Pascal (here real-valued signal not complex)
        num_samples=512,                    # fast time samples per chirp
        rx_mask=(1 << rx_antenna_idx),      # activate only antenna with rx_antenna_idx
        tx_mask=1,                          # TX antenna 1 activated (we only have one tx)
        tx_power_level=31,                  # 
        lp_cutoff_Hz=int(1280000/2)*2,      # Anti-aliasing cutoff frequency
        hp_cutoff_Hz=80000,                 # cutoff frequency for high-pass filter
        if_gain_dB=33))
    
    sequence = device.create_simple_sequence(config)
    device.set_acquisition_sequence(sequence)
    
    return device

def range_Doppler(data):
    # adapted from Infineon examples:
    # /home/audiolab/Documents/radar_sdk_new/examples/py/BGT60TR13C/helpers/Doppler/Algo.py
    # Takes as input one data cube of shape (n_chirps, n_samples_per_chirp)
    num_chirps_per_frame = data.shape[0]
    num_samples = data.shape[1]

    # compute Blackman-Harris Window matrix over chirp samples(range)
    range_window = signal.windows.blackmanharris(num_samples).reshape(1, num_samples)

    # compute Blackman-Harris Window matrix over number of chirps(velocity)
    doppler_window = signal.windows.blackmanharris(num_chirps_per_frame).reshape(1, num_chirps_per_frame)

    # Step 1 - Remove average from signal (mean removal)
    data = data - np.average(data)

    # Step 2 was MTI --> we can't filter out the needed zero-Doppler targets --> skip
    
    # Step 3 - calculate fft spectrum for the frame
    # fft1d = fft_spectrum(data_mti, self.range_window)
    fft1d = fft_spectrum(data, range_window)                  

    # prepare for doppler FFT

    # Transpose
    # Distance is now indicated on y axis
    fft1d = np.transpose(fft1d)

    # Step 4 - Windowing the Data in doppler
    fft1d = np.multiply(fft1d, doppler_window)

    # zp2 = np.pad(fft1d, ((0, 0), (0, num_chirps_per_frame)), "constant")
    zp2 = fft1d
    fft2d = np.fft.fft(zp2) #/ num_chirps_per_frame
    # re-arrange fft result for zero speed at centre
    return np.fft.fftshift(fft2d, (1,))

def record_radar(x, y, n_frames, n_times, path):
    """
    Records a number of n_frames raw radar cubes for position x and y.
    
    x : float
        The x coordinate of the radar.
    y : float
        The y coordinate of the radar.
    n_frames : int
        The number of frames.
    n_times : int
        The number of times to repeat the measurement.
    
    x and y are given in m and single digit millimeter precision 
    
    Example: record_radar(12.5, 25.0, 10) # change later, this is not mm precision
    
    """
    
    # device = config_device_from_metrics()[0]
    device = init_radar_from_config()

    for j in tqdm(range(n_times), leave=False):
        folder_path = path + str(j)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        for i in range(n_frames):
            frame = get_radar_cube(device)
            with open(folder_path + '/' + str(i) + '.pickle', "wb") as output_file:
                pickle.dump(frame, output_file)
        time.sleep(1)

    
def open_radar(x, idx, folder_path):
    """
    Opens the previously recorded data to process.

    Parameters
    ----------
    
    x : float
        The x coordinate of the radar.
    y : float
        The y coordinate of the radar.
    idx : int
        The index of the data to load (we can have multiple measurements
                                       of the same position to increase 
                                       the variance of the data.)
    
    Returns
    -------
    radar_cube.

    """
    folder_path = folder_path + str(x)
    with open(folder_path +'/'+str(idx)+'.pickle', 'rb') as f:
        radar_cube = pickle.load(f)
        
    return radar_cube


def plot_range_doppler(rD, range_res, vel_res, scale='linear', cfar=None, fs=24):
    """
    Plots the Range-Doppler power spectrum.

    Parameters:
    rD : np.ndarray
        2D float array of shape (fast time samples, slow time samples). The
        range-Doppler map in linear power scale.
    range_res : float
        The radar range resolution (m).
    vel_res : float
        The radar velocity resolution (m/s)
    scale : string
        'linear' or 'log' scale for plotting. Both plots are ~ power.
    fs : int
        Font size
    """

    num_samples, num_chirps = rD.shape

    # Define axes for the range and velocity
    X = np.arange(0., num_samples * range_res, range_res)
    Y = np.arange(-vel_res * num_chirps / 2., vel_res * num_chirps / 2., vel_res)

    # Create the power spectrum plot
    plt.figure(figsize=(10, 6))

    if scale == 'log':
        C = 10 * np.log10(rD)  # Log scale (dB)
        plt.title('Range-Doppler power spectrum', fontsize=fs)
    else:
        C = rD  # Linear scale
        plt.title('Range-Doppler power spectrum', fontsize=fs)

    # Plot Range-Doppler power spectrum
    plt.pcolormesh(Y, X, C, shading='auto')
    cbar = plt.colorbar()

    cbar.set_label(label='Power (dB)' if scale == 'log' else 'Power (linear)',
                   fontsize=fs - 8)
    cbar.ax.tick_params(labelsize=fs - 8)
    # Axis labels and formatting with larger fonts
    plt.xlabel('Velocity (m/s)', fontsize=fs - 4)
    plt.ylabel('Range (m)', fontsize=fs - 4)
    plt.grid(True)

    plt.xticks(fontsize=fs - 8)
    plt.yticks(fontsize=fs - 8)
    plt.tight_layout()

    plt.show()
    time.sleep(0.1)
    plt.close('all')


def plot_cfar(cfar, range_res, vel_res, fs):
    """

    Parameters
    ----------
    cfar: np.ndarray
        2D binary array of shape (fast time samples, slow time samples)
    range_res : float
        The radar range resolution (m).
    vel_res : float
        The radar velocity resolution (m/s).
    fs = int
        Font size.
    Returns
    -------
    None.

    """
    num_samples, num_chirps = cfar.shape

    # Define axes for the range and velocity
    X = np.arange(0., num_samples * range_res, range_res)
    Y = np.arange(-vel_res * num_chirps / 2., vel_res * num_chirps / 2., vel_res)

    # Set the boundaries and normalization to map 0 to background, 1 to foreground
    norm = BoundaryNorm(boundaries=[-.5, 0.5, 1.5], ncolors=2)
    cmap = ListedColormap(['white', 'black'])

    # Plotting the matrix with pcolormesh using the custom colormap
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(Y, X, cfar, cmap=cmap, norm=norm)
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.tick_params(labelsize=fs - 8)
    cbar.ax.set_yticklabels(['No target', 'Target'])  # Set the labels
    plt.grid(True)
    plt.title('CFAR target mask', fontsize=fs)
    plt.xlabel('Velocity (m/s)', fontsize=fs - 4)
    plt.ylabel('Range (m)', fontsize=fs - 4)
    plt.xticks(fontsize=fs - 8)
    plt.yticks(fontsize=fs - 8)
    plt.tight_layout()

    plt.show()
    time.sleep(0.1)
    plt.close('all')


def plots(rD, cfar):
    """
    Quick helper to plot rD map, rD map in dB, cfar mask etc.

    Parameters
    ----------
    rD : TYPE
        DESCRIPTION.
    cfar : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    n_samples = 256
    range_res = 0.075
    n_chirps = 32
    vel_res = .2
    
    X = np.arange(0., n_samples*range_res, range_res)
    Y = np.arange(-vel_res * n_chirps / 2., vel_res * n_chirps / 2., vel_res)
    C=abs(rD)**2
    plt.figure()
    plt.pcolormesh(Y, X, C)
    
    
                   
    plt.figure()
    C = 10*np.log10(abs(rD)**2)
    plt.pcolormesh(Y, X, C)
    
    
    plt.figure()
    plt.pcolormesh(Y, X, cfar)


# Record and save dataset
# =============================================================================
if __name__ == '__main__':

    time.sleep(20)

    for i in tqdm(range(1000), desc='Outer loop', colour='red', leave=True):

        # Log the information during the recording, such as room, status, time, etc.
        room_id = 'Corridor'    # The room id contains [softwareLab1, softwareLab2, Corridor, etc.]
        status = 'movingRadar'     # The status contains [static, movingPerson, movingRadar]
        t = time.localtime()
        date = str(t.tm_year) + str(t.tm_mon) + str(t.tm_mday)
        info = room_id + '_' + status + '_' + str(t.tm_year) + '_' + str(t.tm_mon) + '_' + str(t.tm_mday) + '_' + str(t.tm_hour) + '_' + str(t.tm_min) + '_' + str(t.tm_sec)

        # Record and store the radar data
        path = '/Users/yinzheming/Downloads/MA/dataset/dataset_2/' + room_id + '/' + info + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        record_radar(0, 0, 8, 5, path)
        print("Interval time to move the radar")
        time.sleep(1)

# # Plot the radar data
# # =============================================================================
# if __name__ == '__main__':
#
#     folder_path = '/Users/yinzheming/Downloads/MA/dataset/dataset_1/20241121/movingRadar/ISScorridor_movingRadar_2024_11_21_15_30_44/'
#     for idx in range(20):
#         cube = open_radar(x=4, idx=idx, folder_path=folder_path)
#
#         ###########################################################################
#         #### Compute and plot a range-Doppler map. ################################
#         ###########################################################################
#
#         # compute range-Doppler map
#         rD = range_Doppler(cube)
#
#         # compute power of range-Doppler map bins
#         rD = abs(rD) ** 2
#
#         # Radar params (range res, velocity res - used for plots)
#         c0 = sp.constants.c  # speed of light
#         B = 2e9  # bandwidth - e.g. 2 GHz
#         range_res = c0 / (2 * B)  # range resolution (~7.5 cm with 2 GHz B)
#
#         fc = 60e9  # carrier frequency - 60 GHz
#         prt = 220e-6  # chirp repetition time (slow time sampling interval)
#         n_chirps = rD.shape[1]  # number of chirps / slow time samples
#         vel_res = c0 / (2 * fc * prt * n_chirps)
#
#         fs = 30     # font size
#
#         # plot power spectrum. we can plot in 'linear' scale (used as cfar input)
#         # or in 'log' scale (often better to visualize in logarithmic scale (decibel - dB)
#         # as this lowers the large dynamic range of the radar data.)
#         plot_range_doppler(rD, range_res, vel_res, scale='log', cfar=None, fs=fs)
#         # plot_range_doppler(rD, range_res, vel_res, scale='linear', cfar=None, fs=fs)



