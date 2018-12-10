import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from tqdm import tqdm
import argparse
import PCArecon as pca

parser = argparse.ArgumentParser()
parser.add_argument('sounddir')
args = parser.parse_args()

sounddir = args.sounddir
n = 6000
num_samples = len(os.listdir(sounddir))
dataset_name = os.path.basename(os.path.normpath(sounddir))
data_matrix = np.zeros((num_samples, n))

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
dataset_file = '{}_audio_matrix.npy'.format(dataset_name)
np.save(dataset_file, data_matrix)
print("Saved matrix of total shape {} at {}".format(data_matrix.shape, dataset_file))
