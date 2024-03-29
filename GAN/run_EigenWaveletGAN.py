import numpy as np
from datetime import datetime
import time
import os
import sys
import csv
import pywt
import librosa
import scipy.io.wavfile as wio
from tqdm import tqdm, trange
from EiGAN import EiGAN
from PCA_eig_recon import get_weights, PCA_recon, wavelet_recon

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import argparse


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class FSDD_EiGAN(object):
    def __init__(self, dataset_file, pcamean, pcacomponents):

        self.pcamean = pcamean
        self.pcacomponents = pcacomponents

        self.dataset_file = dataset_file

        xd = np.load(dataset_file)
        self.num_components = xd.shape[1]
        self.EiGAN = EiGAN(num_components=self.num_components)
        self.discriminator =  self.EiGAN.discriminator_model()
        self.adversarial = self.EiGAN.adversarial_model()
        self.generator = self.EiGAN.generator()

    def train_GAN(self, num_epochs=100, batch_size=32, img_interval = 10, patience=10, fs=44100):
        
        datestr = "{:%m%d%y_%H%M%S}".format(datetime.now())
        run_directory = '{}_runs/{}/'.format(self.dataset_file, datestr)
        os.makedirs(run_directory, exist_ok=True)
        model_dir = os.path.join(run_directory, 'discriminator_models')
        os.makedirs(model_dir, exist_ok=True)

        x_data = np.load(self.dataset_file)
        self.fs = fs
        print("============================================\r\n======================================================\r\n")
        print("x_data: {}".format(x_data.shape))
        #  y_labels = np.ones((len(x_data)))
        positive_y = np.ones((batch_size,1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
        num_samples = len(x_data)

        fields = ['epoch', 'd_loss', 'd_acc', 'a_loss', 'a_acc']
        with open(os.path.join(run_directory, 'log.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

        last_a_loss = float('inf')
        patience_counter = 0
        displayed_samples = None

        #  noise_vector = np.random.uniform(-1.0, 1.0, size=(1, 100))
        #  fake_im = self.generator.predict(noise_vector)
        #  wavelet = fake_im[0]
        #  cA, cD = wavelet[0,:], wavelet[1,:]
        #  print("wavelet: {}, {}".format(cA.shape, cD.shape))

        for epoch in range(num_epochs):
            print("Epoch {}".format(epoch))

            np.random.shuffle(x_data)

            d_loss_total = [0.0, 0.0]
            a_loss_total = [0.0, 0.0]
            starttime = datetime.now()

            # Get Batch
            pbar = trange(num_samples // batch_size)
            for batch_num in pbar:

                minibatch = x_data[batch_num*batch_size:(batch_num+1)*batch_size]
                noise = np.random.uniform(-1.0, 1.0, (batch_size, 100)).astype(np.float32)
                d_loss = self.discriminator.train_on_batch([minibatch, noise], [positive_y, negative_y, dummy_y])
                a_loss = self.adversarial.train_on_batch(np.random.uniform(-1.0, 1.0, (batch_size, 100)), positive_y)
                a_loss = self.adversarial.train_on_batch(np.random.uniform(-1.0, 1.0, (batch_size, 100)), positive_y)

                # Report Loss and Accuracy
                d_loss_total[0] += d_loss[0]
                #  cur_acc = (d_loss[4] + d_loss[5])/2
                #  d_loss_total[1] = ((d_loss_total[1]*batch_size*batch_num) + cur_acc*batch_size)/((batch_num+1)*batch_size)
                a_loss_total[0] += a_loss[0]
                a_loss_total[1] = ((a_loss_total[1]*batch_size*batch_num) + a_loss[1]*batch_size)/((batch_num+1)*batch_size)
                pbar.set_description("A_acc: {:.3f}".format(a_loss_total[1]))

            endtime = datetime.now()
            print("    epoch time: {}".format(endtime-starttime))

            #  print("    d_loss: {}".format(d_loss_total))
            print("    a_loss: {}".format(a_loss_total))

            displayed_samples = self.generator.predict(np.random.uniform(-1.0, 1.0, (1, 100)))

            row = [epoch, d_loss_total[0], d_loss_total[1], a_loss_total[0], a_loss_total[1]]
            with open(os.path.join(run_directory, 'log.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)


            if epoch%img_interval == 0:
                self.adversarial.save(os.path.join(model_dir, 'adversarial_checkpoint_acc{}_e{}.h5'.format(a_loss_total[1], epoch)))                
                reconstruction = PCA_recon(displayed_samples[0], self.pcamean, self.pcacomponents)
                reconstruction = wavelet_recon(reconstruction)
                librosa.output.write_wav(os.path.join(run_directory, "reconstruction_e{}_normalized.wav".format(epoch)), reconstruction, self.fs, norm=True)
                librosa.output.write_wav(os.path.join(run_directory, "reconstruction_e{}.wav".format(epoch)), reconstruction, self.fs, norm=False)

            if a_loss_total[0] >= last_a_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                last_a_loss = a_loss_total[0]

            if patience_counter >= patience:
                print("Adversarial loss did not improve for {} epochs. Stopping early at epoch {}...".format(patience, epoch))
                break

        self.adversarial.save(os.path.join(model_dir, 'adversarial_final_acc{}.h5'.format(a_loss_total[1])))                
        reconstruction = PCA_recon(displayed_samples[0], self.pcamean, self.pcacomponents)
        reconstruction = wavelet_recon(reconstruction)
        #  wio.write(os.path.join(run_directory, "reconstruction_final.wav"), self.fs, reconstruction)
        librosa.output.write_wav(os.path.join(run_directory, "reconstruction_final_normalized.wav"), reconstruction, self.fs, norm=True)
        librosa.output.write_wav(os.path.join(run_directory, "reconstruction_final.wav"), reconstruction, self.fs, norm=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file')
    parser.add_argument('pca_file')
    parser.add_argument('--fs', type=int, default=44100, help='Sampling rate dataset was created at')
    args = parser.parse_args()
    npz = np.load(args.pca_file)
    pca_mean = npz['mean']
    pca_components = npz['components']
    waveletGAN = FSDD_EiGAN(args.dataset_file, pca_mean, pca_components)
    timer = ElapsedTimer()
    waveletGAN.train_GAN(num_epochs=500, batch_size=16, img_interval=10, patience=500, fs=args.fs)
    timer.elapsed_time()
