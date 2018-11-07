
# coding: utf-8

# In[1]:




import argparse
from pylab import *
import os
import imageio
import audio_utilities
from PIL import Image
import librosa

# Author: Brian K. Vogel
# brian.vogel@gmail.com
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def run_recon():
    """Test Griffin & Lim method for reconstructing audio from a magnitude spectrogram.

        Example of using the Griffin-Lim algorithm. The input file is loaded, the
        spectrogram is computed (note that we discard the phase information). Then,
        using only the (magnitude) spectrogram, the Griffin-Lim algorithm is run
        to reconstruct an audio signal from the spectrogram. The reconstructed audio
        is finally saved to a file.

        A plot of the spectrogram is also displayed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default="bkvhi.wav",
                        help='Input WAV file')
    parser.add_argument('--sample_rate_hz', default=44100, type=int,
                        help='Sample rate in Hz')
    parser.add_argument('--fft_size', default=2048, type=int,
                        help='FFT siz')
    parser.add_argument('--iterations', default=300, type=int,
                        help='Number of iterations to run')
    parser.add_argument('--enable_filter', action='store_true',
                        help='Apply a low-pass filter')
    parser.add_argument('--enable_mel_scale', action='store_true',
                        help='Convert to mel scale and back')
    parser.add_argument('--cutoff_freq', type=int, default=1000,
                        help='If filter is enable, the low-pass cutoff frequency in Hz')
    args = parser.parse_args()

    in_file = Image.open(args.in_file)

    #print(in_file.shape)
    in_file = in_file.resize((1025, 640), Image.ANTIALIAS)
    ext = ".png"
    in_file.save("rescaledimage" + ext)
    in_file = plt.imread("rescaledimage.png")
    print(in_file.shape)
    in_file = rgb2gray(in_file)
    hopsamp = args.fft_size // 8
    print(in_file.shape)
    # Use the Griffin&Lim algorithm to reconstruct an audio signal from the
    # magnitude spectrogram.
    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(in_file,
                                                                   args.fft_size, hopsamp,
                                                                   args.iterations)

    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:
        x_reconstruct = x_reconstruct / max_sample

    # Save the reconstructed signal to a WAV file.
    audio_utilities.save_audio_to_file(x_reconstruct, args.sample_rate_hz)



if __name__ == '__main__':
    run_recon()
