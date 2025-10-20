import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gin
from scipy import signal
import pickle
import io
from PIL import Image
import warnings


@gin.configurable
def resolution_calculation(start_frequency_GHz=59, end_frequency_GHz=63, chirp_repetition_time_s=0.000220, num_chirps=64):
    range_resolution = 3 * 10**8 / (2 * (end_frequency_GHz - start_frequency_GHz) * 10**9)
    velocity_resolution = 3 * 10**8 / (2 * start_frequency_GHz * 10**9) / (chirp_repetition_time_s * num_chirps)

    return range_resolution, velocity_resolution


def figure_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    img = Image.open(buf)
    img_array = np.array(img)

    buf.close()

    return img_array


@gin.configurable
def plot_range_doppler(rD, range_res, vel_res, data_type, range_limit=True, scale='linear', cfar=None, fs=24, R=50, vmin=-80, vmax = 60):
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
    R : float, by default 50 Ohm
    """

    num_samples, num_chirps = rD.shape

    # Define axes for the range and velocity
    X = np.arange(0., num_samples * range_res, range_res)
    Y = np.arange(-vel_res * num_chirps / 2., vel_res * num_chirps / 2., vel_res)

    # Create the power spectrum plot
    fig = plt.figure(figsize=(10, 6))

    if scale == 'log':
        C = 10 * np.log10(rD)  # Log scale (dB)
        plt.title(data_type+' Range-Doppler Spectrum', fontsize=fs)
    else:
        C = rD  # Linear scale
        # C = tf.math.pow(rD, 2) / R
        plt.title(data_type+' Range-Doppler Spectrum', fontsize=fs)

    max_value = np.max(C)
    min_value = np.min(C)
    if max_value > vmax or min_value < vmin:
        warnings.warn(f"The value range of the {data_type} image is over the range of the default colorbar range.")

    # Plot Range-Doppler power spectrum
    if range_limit:
        plt.pcolormesh(Y, X, C, shading='auto', vmin=vmin, vmax=vmax)
    else:
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

    plt.close()

    return fig


def synthesis_12_super_rd(super_res1, super_res2, range_res, velocity_res, scale='linear'):
    '''Synthesis 1x2 super-resolution range-Doppler maps'''
    scale_type = 'log'
    fig_super_res1 = plot_range_doppler(abs(super_res1)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res2 = plot_range_doppler(abs(super_res2)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)

    fig_size = fig_super_res1.get_size_inches() # (width, height)
    large_fig = plt.figure(figsize=(8,6))

    ax1 = large_fig.add_subplot(121)
    ax1.imshow(figure_to_png(fig_super_res1))
    ax1.axis('off')

    ax2 = large_fig.add_subplot(122)
    ax2.imshow(figure_to_png(fig_super_res2))
    ax2.axis('off')

    large_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0)

    plt.show()


def synthesis_13_super_rd(super_res1, super_res2, super_res3, range_res, velocity_res, scale='linear'):
    '''Synthesis 1x3 super-resolution range-Doppler maps'''
    scale_type = 'log'
    fig_super_res1 = plot_range_doppler(abs(super_res1)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res2 = plot_range_doppler(abs(super_res2) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res3 = plot_range_doppler(abs(super_res3)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)

    fig_size = fig_super_res1.get_size_inches() # (width, height)
    large_fig = plt.figure(figsize=(10,6))

    ax1 = large_fig.add_subplot(131)
    ax1.imshow(figure_to_png(fig_super_res1))
    ax1.axis('off')

    ax2 = large_fig.add_subplot(132)
    ax2.imshow(figure_to_png(fig_super_res2))
    ax2.axis('off')

    ax3 = large_fig.add_subplot(133)
    ax3.imshow(figure_to_png(fig_super_res3))
    ax3.axis('off')

    large_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0)

    plt.show()


def synthesis_13_rd(low_res, super_res, high_res, range_res, velocity_res, scale='linear'):
    '''Synthesis 1x3 range-Doppler maps'''
    scale_type = 'log'
    fig_low_res = plot_range_doppler(abs(low_res)**2, range_res*2, velocity_res*2, data_type='Low resolutional', scale=scale_type)
    fig_super_res = plot_range_doppler(abs(super_res)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_high_res = plot_range_doppler(abs(high_res)**2, range_res, velocity_res, data_type='High resolutional', scale=scale_type)

    fig_size = fig_high_res.get_size_inches() # (width, height)
    large_fig = plt.figure(figsize=(10,6))

    ax1 = large_fig.add_subplot(131)
    ax1.imshow(figure_to_png(fig_low_res))
    ax1.axis('off')

    ax2 = large_fig.add_subplot(132)
    ax2.imshow(figure_to_png(fig_super_res))
    ax2.axis('off')

    ax3 = large_fig.add_subplot(133)
    ax3.imshow(figure_to_png(fig_high_res))
    ax3.axis('off')

    large_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0)

    plt.show()


def synthesis_22_super_rd(super_res1, super_res2, super_res3, super_res4, range_res, velocity_res, scale='linear'):
    '''Synthesis 2x2 super-resolution range-Doppler maps'''
    scale_type = 'log'
    fig_super_res1 = plot_range_doppler(abs(super_res1)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res2 = plot_range_doppler(abs(super_res2) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res3 = plot_range_doppler(abs(super_res3) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res4 = plot_range_doppler(abs(super_res4) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)

    fig_size = fig_super_res1.get_size_inches() # (width, height)
    large_fig = plt.figure(figsize=(10,6))

    ax1 = large_fig.add_subplot(221)
    ax1.imshow(figure_to_png(fig_super_res1))
    ax1.axis('off')

    ax2 = large_fig.add_subplot(222)
    ax2.imshow(figure_to_png(fig_super_res2))
    ax2.axis('off')

    ax3 = large_fig.add_subplot(223)
    ax3.imshow(figure_to_png(fig_super_res3))
    ax3.axis('off')

    ax4 = large_fig.add_subplot(224)
    ax4.imshow(figure_to_png(fig_super_res4))
    ax4.axis('off')

    large_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0)

    plt.show()


def synthesis_22_rd(low_res, high_res, super_res1, super_res2, range_res, velocity_res, scale='linear'):
    '''Synthesis 2x2 range-Doppler maps'''
    scale_type = 'log'
    fig_low_res = plot_range_doppler(abs(low_res) ** 2, range_res*2, velocity_res*2, data_type='Low resolutional',scale=scale_type)
    fig_high_res = plot_range_doppler(abs(high_res) ** 2, range_res, velocity_res, data_type='High resolutional',scale=scale_type)
    fig_super_res1 = plot_range_doppler(abs(super_res1)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res2 = plot_range_doppler(abs(super_res2) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)

    fig_size = fig_super_res1.get_size_inches() # (width, height)
    large_fig = plt.figure(figsize=(10,6))

    ax1 = large_fig.add_subplot(221)
    ax1.imshow(figure_to_png(fig_low_res))
    ax1.axis('off')

    ax2 = large_fig.add_subplot(222)
    ax2.imshow(figure_to_png(fig_high_res))
    ax2.axis('off')

    ax3 = large_fig.add_subplot(223)
    ax3.imshow(figure_to_png(fig_super_res1))
    ax3.axis('off')

    ax4 = large_fig.add_subplot(224)
    ax4.imshow(figure_to_png(fig_super_res2))
    ax4.axis('off')

    large_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0)

    plt.show()


def synthesis_23_super_rd(super_res1, super_res2, super_res3, super_res4, super_res5, super_res6, range_res, velocity_res, scale='linear'):
    '''Synthesis 2x3 super-resolutiono range-Doppler maps'''
    scale_type = 'log'
    fig_super_res1 = plot_range_doppler(abs(super_res1)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res2 = plot_range_doppler(abs(super_res2) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res3 = plot_range_doppler(abs(super_res3) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res4 = plot_range_doppler(abs(super_res4) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res5 = plot_range_doppler(abs(super_res5) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_super_res6 = plot_range_doppler(abs(super_res6) ** 2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)

    fig_size = fig_super_res1.get_size_inches() # (width, height)
    large_fig = plt.figure(figsize=(10, 4))

    ax1 = large_fig.add_subplot(231)
    ax1.imshow(figure_to_png(fig_super_res1))
    ax1.axis('off')

    ax2 = large_fig.add_subplot(232)
    ax2.imshow(figure_to_png(fig_super_res2))
    ax2.axis('off')

    ax3 = large_fig.add_subplot(233)
    ax3.imshow(figure_to_png(fig_super_res3))
    ax3.axis('off')

    ax4 = large_fig.add_subplot(234)
    ax4.imshow(figure_to_png(fig_super_res4))
    ax4.axis('off')

    ax5 = large_fig.add_subplot(235)
    ax5.imshow(figure_to_png(fig_super_res5))
    ax5.axis('off')

    ax6 = large_fig.add_subplot(236)
    ax6.imshow(figure_to_png(fig_super_res6))
    ax6.axis('off')

    large_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0)

    plt.show()


def synthesis_23_rd(low_res, super_res, high_res, super_benchmark_res, range_res, velocity_res, scale='linear'):
    '''Synthesis 2x3 range-Doppler maps'''
    scale_type = 'log'
    fig_low_res = plot_range_doppler(abs(low_res)**2, range_res*2, velocity_res*2, data_type='Low resolutional', scale=scale_type)
    fig_super_res = plot_range_doppler(abs(super_res)**2, range_res, velocity_res, data_type='Super resolutional', scale=scale_type)
    fig_high_res = plot_range_doppler(abs(high_res)**2, range_res, velocity_res, data_type='High resolutional', scale=scale_type)
    fig_super_benchmark_res = plot_range_doppler(abs(super_benchmark_res)**2, range_res, velocity_res, data_type='Benchmark super resolutional', scale=scale_type)

    fig_size = fig_high_res.get_size_inches() # (width, height)
    large_fig = plt.figure(figsize=(10,6))

    ax1 = large_fig.add_subplot(231)
    ax1.imshow(figure_to_png(fig_low_res))
    ax1.axis('off')

    ax2 = large_fig.add_subplot(232)
    ax2.imshow(figure_to_png(fig_super_res))
    ax2.axis('off')

    ax3 = large_fig.add_subplot(233)
    ax3.imshow(figure_to_png(fig_high_res))
    ax3.axis('off')

    ax4 = large_fig.add_subplot(234)
    ax4.imshow(figure_to_png(fig_low_res))
    ax4.axis('off')

    ax5 = large_fig.add_subplot(235)
    ax5.imshow(figure_to_png(fig_super_benchmark_res))
    ax5.axis('off')

    ax6 = large_fig.add_subplot(236)
    ax6.imshow(figure_to_png(fig_high_res))
    ax6.axis('off')

    large_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0)

    plt.show()


def synthesis_33_rd(low_res, high_res, super_res_1, super_res_2, super_res_3, super_res_4, super_res_5, super_res_6, super_res_7, range_res, velocity_res):
    '''Synthesis 3x3 range-Doppler maps'''
    scale_type = 'log'
    fig_low_res = plot_range_doppler(abs(low_res)**2, range_res*2, velocity_res*2, data_type='Low resolutional', scale=scale_type)
    fig_high_res = plot_range_doppler(abs(high_res)**2, range_res, velocity_res, data_type='High resolutional', scale=scale_type)
    fig_super_res_1 = plot_range_doppler(abs(super_res_1) ** 2, range_res, velocity_res, data_type='Interpolation super resolutional', scale=scale_type)
    fig_super_res_2 = plot_range_doppler(abs(super_res_2) ** 2, range_res, velocity_res, data_type='CNN_simple super resolutional', scale=scale_type)
    fig_super_res_3 = plot_range_doppler(abs(super_res_3) ** 2, range_res, velocity_res, data_type='UNet_simple super resolutional', scale=scale_type)
    fig_super_res_4 = plot_range_doppler(abs(super_res_4) ** 2, range_res, velocity_res, data_type='UNet_concat super resolutional', scale=scale_type)
    fig_super_res_5 = plot_range_doppler(abs(super_res_5) ** 2, range_res, velocity_res, data_type='DP super resolutional', scale=scale_type)
    fig_super_res_6 = plot_range_doppler(abs(super_res_6) ** 2, range_res, velocity_res, data_type='SwinIR+DP super resolutional', scale=scale_type)
    fig_super_res_7 = plot_range_doppler(abs(super_res_7) ** 2, range_res, velocity_res, data_type='SwinIR+Swin super resolutional', scale=scale_type)

    fig_size = fig_high_res.get_size_inches() # (width, height)
    large_fig = plt.figure(figsize=(10,6))

    ax1 = large_fig.add_subplot(331)
    ax1.imshow(figure_to_png(fig_low_res))
    ax1.axis('off')

    ax2 = large_fig.add_subplot(332)
    ax2.imshow(figure_to_png(fig_high_res))
    ax2.axis('off')

    ax3 = large_fig.add_subplot(333)
    ax3.imshow(figure_to_png(fig_super_res_1))
    ax3.axis('off')

    ax4 = large_fig.add_subplot(334)
    ax4.imshow(figure_to_png(fig_super_res_2))
    ax4.axis('off')

    ax5 = large_fig.add_subplot(335)
    ax5.imshow(figure_to_png(fig_super_res_3))
    ax5.axis('off')

    ax6 = large_fig.add_subplot(336)
    ax6.imshow(figure_to_png(fig_super_res_4))
    ax6.axis('off')

    ax7 = large_fig.add_subplot(337)
    ax7.imshow(figure_to_png(fig_super_res_5))
    ax7.axis('off')

    ax8 = large_fig.add_subplot(338)
    ax8.imshow(figure_to_png(fig_super_res_6))
    ax8.axis('off')

    ax9 = large_fig.add_subplot(339)
    ax9.imshow(figure_to_png(fig_super_res_7))
    ax9.axis('off')

    large_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0)

    plt.show()



if __name__ == "__main__":
    rD_path = '/Users/yinzheming/Downloads/MA/dataset_sample/20241016/movingPerson/softwareLab2_movingPerson_2024_10_16_15_42_1/7/7.pickle'
    rD = open(rD_path,'rb')
    rD=pickle.load(rD)
    print(rD.shape)
    print(rD.dtype)

    rD = range_Doppler(rD)
    print(rD.shape)
    print(rD.dtype)

    # rD = rD.astype(np.complex64)
    # rD = np.fft.fft2(rD.transpose(1, 0))
    # rD = np.fft.fftshift(rD, axes=1)
    # print(rD.shape)
    # print(rD.dtype)

    rD_power = abs(rD) ** 2
    print(rD_power.shape)
    print(rD_power.dtype)

    range_res, velocity_res = resolution_calculation()

    fig = plot_range_doppler(rD_power, range_res, velocity_res, scale='linear')
