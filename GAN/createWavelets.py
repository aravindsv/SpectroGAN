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
parser.add_argument('-s', '--samples', help='Samples per training instance', default=64, type=int)
args = parser.parse_args()

sounddir = args.sounddir
samples_per_ti = args.samples
dataset_name = os.path.basename(os.path.normpath(sounddir))

test, _ = librosa.core.load(os.path.join(sounddir, os.listdir(sounddir)[0]), sr=None)
cat, cdt = pywt.dwt(test, 'db2')
wavelet_len = (len(cat)//4)*4

num_training_instances = int(np.round(len(os.listdir(sounddir))//samples_per_ti))

wavelet_arr = np.zeros((num_training_instances, 2*samples_per_ti, wavelet_len))
cur_ti = np.zeros((2*samples_per_ti, wavelet_len))

i = 0
j = 0
for soundfile in tqdm(os.listdir(sounddir)):
    wav, fs = librosa.core.load(os.path.join(sounddir, soundfile), sr=None)
    #  fs, wav = wio.read(os.path.join(sounddir, soundfile))
    cA, cD = pywt.dwt(wav, 'db2')
    cA, cD = cA[:wavelet_len], cD[:wavelet_len]
    waveletc = np.vstack([cA, cD])
    #  waveletc = waveletc.reshape(210,210,1)
    #  wavelet_arr.append(waveletc)
    cur_ti[j*2] = cA
    cur_ti[j*2 + 1] = cD
    j += 1
    if j >= samples_per_ti:
        wavelet_arr[i] = cur_ti
        i += 1
        j = 0
        cur_ti = np.zeros((2*samples_per_ti, wavelet_len))
        

#  wavelet_arr = np.array(wavelet_arr)
wavelet_dataset_file = '{}_wavelets.npy'.format(dataset_name)
np.save(wavelet_dataset_file, wavelet_arr)
print("Saved wavelets of total shape {} at {}".format(wavelet_arr.shape, wavelet_dataset_file))
