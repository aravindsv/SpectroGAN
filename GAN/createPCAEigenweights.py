import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from tqdm import tqdm
import argparse
from PCArecon import get_weights

parser = argparse.ArgumentParser()
parser.add_argument('sounddir')
parser.add_argument('eigendir')
args = parser.parse_args()

sounddir = args.sounddir
eigendir = args.eigendir
dataset_name = os.path.basename(os.path.normpath(sounddir))
data_matrix = []
matrix = np.load(eigendir)
mean = matrix['mean']
eigen = matrix['components']
sample_len = 44100

for soundfile in tqdm(os.listdir(sounddir)):
    wav, fs = librosa.core.load(os.path.join(sounddir, soundfile), sr=None)
    wav = np.array(wav)
    if len(wav) != sample_len:
        continue
    weights_array = get_weights(wav, mean, eigen)
    data_matrix.append(weights_array)

data_matrix = np.array(data_matrix)
dataset_file = '{}_eigaudio_matrix.npy'.format(dataset_name)
np.save(dataset_file, data_matrix)
print("Saved data_matrix of shape {} to {}".format(data_matrix.shape, dataset_file))
