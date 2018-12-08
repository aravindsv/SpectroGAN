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
dataset_name = os.path.basename(os.path.normpath(sounddir))
data_matrix = []

for soundfile in tqdm(os.listdir(sounddir)):
    wav, fs = librosa.core.load(os.path.join(sounddir, soundfile), sr=None)
    data_matrix = np.hstack(data_matrix, wav)

data_matrix = np.array(data_matrix)
dataset_file = '{}_audio_matrix.npy'.format(dataset_name)
np.save(dataset_file, data_matrix)
print("Saved matrix of total shape {} at {}".format(data_matrix.shape, dataset_file))
