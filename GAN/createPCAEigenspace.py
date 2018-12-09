import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from tqdm import tqdm
import argparse
import PCArecon as pca

n = 44100

parser = argparse.ArgumentParser()
parser.add_argument('sounddir')
args = parser.parse_args()

sounddir = args.sounddir
n = 44100
num_samples = len(os.listdir(sounddir))
dataset_name = os.path.basename(os.path.normpath(sounddir))
num_clips = len(os.listdir(sounddir))
data_matrix = np.zeros((num_clips, n))


i = 0
for soundfile in tqdm(os.listdir(sounddir)):
    wav, fs = librosa.core.load(os.path.join(sounddir, soundfile), sr=None)
    wav = np.array(wav)
    if len(wav) != n:
        continue
    data_matrix[i,:] = wav
    i +=1

data_matrix = data_matrix[:i]

#data_matrix = np.array(data_matrix)
dataset_file = '{}_eigaudio_matrix.npy'.format(dataset_name)
np.save(dataset_file, data_matrix)
print("Saved matrix of total shape {} at {}".format(data_matrix.shape, dataset_file))
