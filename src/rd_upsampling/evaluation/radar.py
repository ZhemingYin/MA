#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May 24 16:18:46 2024

Script to setup radar and fetch single frames from one antenna. This also includes
both range- and range-Doppler fft.

The radar python wrapper (ifxradarsdk) is installed on both devices (PC and Raspi).
On the Raspberry this only works with 32 bit Raspberry Pi OS  -->
No Ubuntu since they stopped 32 bit support with version 16 or 18. 

The issue is that pywheels (the library required to build the python wrapper)
doesn't support a 64 bit OS right now: https://blog.piwheels.org/raspberry-pi-os-64-bit-aarch64/'.
There is no workaround (except maybe using something like Ubuntu in Docker or an Ubuntu VM or switching to a 64-bit mini pc).

This is mostly copied from the given example code by Infineon: 
    
~/Documents/radar_sdk_new/examples/py/BGT60TR13C/raw_data.py
~/Documents/radar_sdk_new/examples/py/BGT60TR13C/distance_fft.py - nvm, this violates Parseval in multiple ways ...

For radar parameter config see Pascals MA, Chenmings PhD or the Radar intro chapter of the
Infineon Radar docs. 
"""

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap

from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics, FmcwSequenceChirp
from scipy import signal
import os
import scipy as sp

# from indoor_positioning.radar.cfar import ca_cfar_2d, os_cfar_2d, \
#     cash_cfar_2d, mamis_cfar_2d, ifar_cfar_2d
#
# from indoor_positioning.radar.clustering import clustering_dbscan, wmean_target_esimate

def get_radar_cube(device):
    # fetches exactly one data cube for one antenna
    
    frame_contents = device.get_next_frame()
    # raw adc data cube for one frame and one antenna
    data_cube = frame_contents[0][0]

    return data_cube

def range_fft(cube, range_window):
    """
    Performs range fft for real-valued (only In-phase channel) baseband signal. 
    This preserves signal energy (Parseval theorem) such that the windowed
    time-domain cube has the same energy as its fft (range-fft) result.

    Parameters
    ----------
    cube : float 
        A 2D (one antenna) baseband data cube, i.e. float array of shape 
        (slow time samples, fast time samples).
    
    range_window : scipy.signal.windows instance
        A window function to window the time-domain samples before fft.

    Returns
    -------
    r_fft : complex
        The range fft output for each chirp. We use an even number of samples
        and have Hermitian symmetry in the spectrum 
        --> shape (slow time samples, fast time samples / 2 + 1)
    """
    [num_chirps, num_samples] = cube.shape
    
    # window the time-domain data
    cube = cube * range_window 
    
    # Compute the FFT (range fft along last axis) of the real-valued (I-channel) baseband signal
    r_fft = np.fft.fft(cube, axis=-1)
    
    # Use only the spectrum for positive frequencies (negative freqs. are redundant - complex conjugate)
    r_fft = r_fft[:,:num_samples // 2 + 1]
    
    # Scale all double (symmetric) frequencies such that the signal energy is preserved.
    # sqrt(2) leads to factor 2 after squaring (to get from voltage to power)
    r_fft[:,1:r_fft.shape[-1]-1] *= np.sqrt(2)
    
    # normalize by sqrt(N) such that we get factor 1/N in power after squaring this later.
    # In fft we sum over N samples, so we need the factor 1/N to normalize in frequency domain after fft.
    r_fft *= 1 / np.sqrt(num_samples)
    
    return r_fft

def init_radar_from_metrics():
    """
    Just some function to set the radar parameters by using the radar metrics.
    Better use init_radar_from_config() to get the exact wanted configuration.

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
        The created FMCW device.
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
    
    rx_antenna_idx = 1
    config = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=1 /10., # fps, not used if data is collected at triggers
    chirp_repetition_time_s=0.000220,    # prt
    num_chirps=32,                          
    tdm_mimo=False,                         # MIMO disabled
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=59_000_000_000,  
        end_frequency_Hz=61_000_000_000,   
        sample_rate_Hz=1_280_000*2,         # We need to double the sample rate from Pascal (here real-valued signal not complex)
        num_samples=512,                    # fast time samples per chirp
        rx_mask=(1 << rx_antenna_idx),      # activate only antenna with rx_antenna_idx https://community.infineon.com/t5/Radar-sensor/RX-mask-setting-of-BGT60TR13C/td-p/669486
        tx_mask=1,                          # TX antenna 1 activated (we only have one tx)
        tx_power_level=31,                  # 
        lp_cutoff_Hz=int(1280000/2)*2,      # Anti-aliasing cutoff frequency
        hp_cutoff_Hz=80000,                 # cutoff frequency for high-pass filter
        if_gain_dB=33))
    
    sequence = device.create_simple_sequence(config)
    device.set_acquisition_sequence(sequence)
    
    return device

def range_Doppler(cube):
    """
    Computes the complex-valued range-Doppler map (2D fft of baseband signal --> still ~ voltage/amplitude U).
    rD input for CFAR is abs(rD)² ~ spectral power P ~ U² : square law detector
    
    Like in the range_fft() function, the energy of the windowed data before
    and after fft is preserved.

    Parameters
    ----------
    cube : np.ndarray
        A 2D (one antenna) baseband data cube, i.e. float array of shape 
        (slow time samples, fast time samples).

    Returns
    -------
    rD : np.ndarray
        A complex valued range-Doppler signal of shape
        (slow time samples, fast time samples / 2 + 1)
        
    """
    
    num_chirps, num_samples = cube.shape

    # compute row (chirp) averages
    avgs = np.average(cube, 1)[:, None]

    # de-bias values --> the fast-time samples per chirp have zero mean
    cube = cube - avgs

    # compute Blackman-Harris window matrix for range processing (fast-time)
    range_window = signal.windows.blackmanharris(num_samples).reshape(1, num_samples)

    # compute Blackman-Harris window matrix for Doppler processing (slow-time)
    doppler_window = signal.windows.blackmanharris(num_chirps).reshape(1, num_chirps)

    # perform range fft for all chirps (fast-time)
    r_fft = range_fft(cube, range_window)                  

    # switch axes for Doppler fft
    r_fft = np.transpose(r_fft)

    # apply window for range-processed Doppler samples
    r_fft = np.multiply(r_fft, doppler_window)
    
    # Doppler fft across all chirps (slow-time) - this has complex-valued inputs --> normal fft
    rD_fft = np.fft.fft(r_fft) / np.sqrt(num_chirps)
    
    # shift frequencies such that zero velocity/Doppler is in center 
    rD_fft = np.fft.fftshift(rD_fft, 1)

    return rD_fft

def record_radar(x, y, n_frames):
    """
    Records a number of n_frames raw radar cubes for position x and y.
    
    x : float
        The x coordinate of the radar.
    y : float
        The y coordinate of the radar.
    n_frames : int
        The number of frames.
    
    x and y are given in m and single digit millimeter precision 
    
    Example: record_radar(12.5, 25.0, 10) # change later, this is not mm precision
    
    """
    folder_path = '../../../data/data_static/'+str(x)+'_'+str(y)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    # device = config_device_from_metrics()[0]
    device = init_radar_from_config()
    
    for i in range(n_frames):
        time.sleep(.1)
        frame = get_radar_cube(device)
        
        with open(folder_path + '/' + str(i) + '.pickle', "wb") as output_file:
            pickle.dump(frame, output_file)
    
def open_radar(x, y, idx):
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
    folder_path = '../../../data/data_static/'+str(x)+'_'+str(y)
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
                       fontsize=fs-8)
    cbar.ax.tick_params(labelsize=fs-8) 
    # Axis labels and formatting with larger fonts
    plt.xlabel('Velocity (m/s)', fontsize=fs-4)
    plt.ylabel('Range (m)', fontsize=fs-4)
    plt.grid(True)
    
    plt.xticks(fontsize=fs-8)
    plt.yticks(fontsize=fs-8)
    plt.tight_layout()
    
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
    cbar.ax.tick_params(labelsize=fs-8) 
    cbar.ax.set_yticklabels(['No target', 'Target'])  # Set the labels
    plt.grid(True)
    plt.title('CFAR target mask', fontsize=fs)
    plt.xlabel('Velocity (m/s)', fontsize=fs-4)
    plt.ylabel('Range (m)', fontsize=fs-4)
    plt.xticks(fontsize=fs-8)
    plt.yticks(fontsize=fs-8)
    plt.tight_layout()
    
def plot_detections(cfar, range_res, vel_res, cluster_cells, cluster_wmeans, fs):
    """
    Plots the clustered rD bins with unique colors for each cluster. Also
    output the target detections (weighted means of clusters) as crosses.
    
    Parameters:
    cfar : np.ndarray
        2D binary array of shape (fast time samples, slow time samples)
        Optional: If cfar is not None, plot a binary CFAR target mask instead of rD.     
    range_res : float
        The radar range resolution (m).
    vel_res : float
        The radar velocity resolution (m/s)
    cluster_cells : list
        The list of cluster cells (rD) bins.
    cluster_wmeans : np.ndarray
        Float array of shape (number clusters, 2) holing the esimated (range, Doppler)
        values of targets.
    fs : int
        font size

    """
    num_samples, num_chirps = cfar.shape
    X = np.arange(0., num_samples * range_res, range_res)
    Y = np.arange(-vel_res * num_chirps / 2., vel_res * num_chirps / 2., vel_res)
    
    # Create an empty grid for visualizing clusters
    cluster_grid = np.full(cfar.shape, -1)  # Initialize to -1 (background/noise)
    num_clusters = len(cluster_cells)
    
    # Assign a unique value for each cluster in the grid
    for cluster_id in range(num_clusters):
            cluster_grid[cluster_cells[cluster_id][:,0], 
                         cluster_cells[cluster_id][:,1]] = cluster_id # Assign IDs starting from 1
    
    # Create color map for clusters (white for background)
    colors = ['white'] + [plt.cm.turbo(i) for i in np.linspace(0., 1, num_clusters)]  # Include white for background
    cmap = ListedColormap(colors)

    # Define boundaries for the discrete colorbar with step size of 1
    bounds = np.arange(-1.5, num_clusters + .5, 1)  # From -0.5 to n_clusters + 0.5 for centering
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

    # Plot the cluster grid
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(Y, X, cluster_grid, shading='auto', cmap=cmap, norm=norm)
    cbar = plt.colorbar(ticks=np.arange(-1, num_clusters ))  # Only add ticks for clusters
    cbar.ax.set_yticklabels(['no target'] + list(range(1, num_clusters + 1)))  # Labels: ['no target', 1, 2, ..., num_clusters]        cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label='Cluster / Target ID', fontsize=fs-8)
    cbar.ax.tick_params(labelsize=fs-8) 
    plt.title('Target detections ', fontsize=fs)
    plt.xlabel('Velocity (m/s)', fontsize=fs-4)
    plt.ylabel('Range (m)', fontsize=fs-4)
    plt.xticks(fontsize=fs-8)
    plt.yticks(fontsize=fs-8)
    
    # Plot the weighted mean target estimates as red 'X' markers
    plt.scatter(cluster_wmeans[:, 1], cluster_wmeans[:, 0], marker='x', color='red', s=100)

    plt.grid(True)
    plt.tight_layout()

if __name__ == '__main__':
    plt.close('all')
    fs = 30 # font size for plots
    
    # plt.close('all')
    ###########################################################################
    #### How to record a 2D baseband data cube (raw radar signal before 2D ####
    ##### range-Doppler FFT) ##################################################
    ###########################################################################
    
    # # set the config, take care of the rx antenna etc. in the config function body
    # device = init_radar_from_config()
    # # trigger a 2D data cube (baseband time-domain signal --> series of sampled chirps)
    # cube = get_radar_cube(device)
    
    ###########################################################################
    #### Compute and plot a range-Doppler map. ################################
    ###########################################################################
    
    # open a previously saved raw radar cube
    cube = open_radar(2.0, 2.0, 1)
    
    # compute range-Doppler map
    rD = range_Doppler(cube)
    
    # compute power of range-Doppler map bins
    rD = abs(rD)**2
    
    # Radar params (range res, velocity res - used for plots)
    c0 = sp.constants.c # speed of light
    B = 2e9 # bandwidth - e.g. 2 GHz
    range_res = c0 / (2*B) # range resolution (~7.5 cm with 2 GHz B)

    fc = 60e9 # carrier frequency - 60 GHz
    prt = 220e-6 # chirp repetition time (slow time sampling interval)
    n_chirps = rD.shape[1] # number of chirps / slow time samples
    vel_res = c0 / (2*fc*prt*n_chirps)
    
    # plot power spectrum. we can plot in 'linear' scale (used as cfar input)
    # or in 'log' scale (often better to visualize in logarithmic scale (decibel - dB) 
    # as this lowers the large dynamic range of the radar data.)
    plot_range_doppler(rD, range_res, vel_res, scale='log', cfar=None, fs=fs)
    plot_range_doppler(rD, range_res, vel_res, scale='linear', cfar=None, fs=fs)
    
    ###########################################################################
    #### Perform cfar detection and plot cfar mask. ###########################
    ###########################################################################
    
    # Perform cfar detection with the given range-Doppler map. 
    # This outputs the binary cfar mask (target / no target).
    # See "cfar.py" for the different variants and hyperparameters (False alarm rate,
    # training cells, guard cells)
    guard_cells = (3, 3)
    training_cells = (8, 8)
    rate_fa = 1e-4
    cfar = ca_cfar_2d(rD, guard_cells = guard_cells, 
                      training_cells= training_cells,
                      rate_fa = rate_fa)
    
    # Plot cfar mask
    plot_cfar(cfar, range_res, vel_res, fs)
    
    ###########################################################################
    #### Cluster the cfar output and perform detection ########################
    ##### with weighted cluster means. ########################################
    ###########################################################################

    # clustering with DBSCAN
    cluster_cells = clustering_dbscan(cfar, eps=1., min_samples = 5)
    
    # target parameter (range, Doppler) estimation with weighted (weights = power) mean 
    cluster_wmeans = wmean_target_esimate(rD, range_res, vel_res, cluster_cells)
    
    # Plot the clusters and detections
    plot_detections(cfar, range_res, vel_res, cluster_cells, cluster_wmeans, fs)