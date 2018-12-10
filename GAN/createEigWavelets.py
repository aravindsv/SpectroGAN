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
n = 6000

sounddir = args.sounddir
dataset_name = os.path.basename(os.path.normpath(sounddir))
#wavelet_arr = []
num_clips = len(os.listdir(sounddir))
wavelet_arr = np.zeros((num_clips, n))
i = 0

for soundfile in tqdm(os.listdir(sounddir)):
    wav, fs = librosa.core.load(os.path.join(sounddir, soundfile), sr=None)
    wav = wav[:n]
    #  fs, wav = wio.read(os.path.join(sounddir, soundfile))
    cA, cD = pywt.dwt(wav, 'db2')
    cA, cD = cA[:-1], cD[:-1]
    waveletc = np.hstack([cA, cD])
    wavelet_arr[i,:] = waveletc
    i+=1

wavelet_arr = np.array(wavelet_arr)
wavelet_dataset_file = '{}_eigenwavelets.npy'.format(dataset_name)
np.save(wavelet_dataset_file, wavelet_arr)
print("Saved wavelets of total shape {} at {}".format(wavelet_arr.shape, wavelet_dataset_file))
