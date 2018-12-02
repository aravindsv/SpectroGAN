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
wavelet_arr = []

for soundfile in tqdm(os.listdir(sounddir)):
    wav, fs = librosa.core.load(os.path.join(sounddir, soundfile), sr=None)
    #  fs, wav = wio.read(os.path.join(sounddir, soundfile))
    cA, cD = pywt.dwt(wav, 'db2')
    N = len(cA)
    waveletc = np.vstack([cA, cD])
    wavelet_arr.append(waveletc.reshape(2,-1,1))

wavelet_arr = np.array(wavelet_arr)
wavelet_dataset_file = '{}_wavelets.npy'.format(dataset_name)
np.save(wavelet_dataset_file, wavelet_arr)
print("Saved wavelets of total shape {} at {}".format(wavelet_arr.shape, wavelet_dataset_file))
