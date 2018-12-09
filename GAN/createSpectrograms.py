import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wio
import librosa
import pywt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sounddir')
args = parser.parse_args()

sounddir = args.sounddir
dataset_name = os.path.basename(os.path.normpath(sounddir))
spectrogram_array = []
avg_phase = None

for idx, soundfile in enumerate(tqdm(os.listdir(sounddir))):
    wav, fs = librosa.core.load(os.path.join(sounddir, soundfile), sr=None)
    spectrogram = librosa.core.stft(wav,
                                    n_fft=2048,
                                    hop_length=256,
                                    window=np.hanning(2048))
    if avg_phase is None:
        avg_phase = np.zeros(spectrogram.shape)
    M = np.abs(spectrogram)
    phase = spectrogram/(M+1e-6)
    avg_phase = (avg_phase*idx + phase)/(idx+1)
    #  phase_array.append(phase)
    M = M.reshape(M.shape[0], M.shape[1], 1)
    spectrogram_array.append(M)

spectrogram_array = np.array(spectrogram_array)
spectrogram_dataset_file = '{}_spectrograms.npy'.format(dataset_name)
np.save(spectrogram_dataset_file, spectrogram_array)
print("Saved spectrograms of total shape {} at {}".format(spectrogram_array.shape, spectrogram_dataset_file))

phase_data_file = '{}_mean_phase.npy'.format(dataset_name)
np.save(phase_data_file, avg_phase)
print("Saved mean phase of all spectrograms at {}".format(phase_data_file))
