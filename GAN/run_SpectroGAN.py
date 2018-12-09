import numpy as np
from datetime import datetime
import time
import os
import sys
import csv
import pywt
import librosa
#  import scipy.io.wavfile as wio
from tqdm import tqdm, trange
from SpectroGAN import SpectroGAN

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

class FSDD_SpectroGAN(object):
    def __init__(self, dataset_file):

        self.dataset_file = dataset_file
        xd = np.load(dataset_file)
        xd0 = xd[0]
        if xd0.shape[0] % 2 != 0:
            xd = xd[:,:-1,:,:]
        self.SpectroGAN = SpectroGAN(img_rows=xd.shape[1], img_cols=xd.shape[2])
        self.discriminator =  self.SpectroGAN.discriminator_model()
        self.adversarial = self.SpectroGAN.adversarial_model()
        self.generator = self.SpectroGAN.generator()

    def train_GAN(self, num_epochs=100, batch_size=32, img_interval = 10, patience=5):
        
        datestr = "{:%m%d%y_%H%M%S}".format(datetime.now())
        run_directory = '{}_runs/{}/'.format(self.dataset_file.split('.')[0], datestr)
        os.makedirs(run_directory, exist_ok=True)
        model_dir = os.path.join(run_directory, 'discriminator_models')
        os.makedirs(model_dir, exist_ok=True)

        x_data = np.load(self.dataset_file)
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
                noise = np.random.uniform(-1.0, 1.0, minibatch.shape).astype(np.float32)
                d_loss = self.discriminator.train_on_batch(np.concatenate([minibatch, noise]), np.concatenate([positive_y, negative_y]))
                a_loss = self.adversarial.train_on_batch(np.random.uniform(-1.0, 1.0, (batch_size, 100)), positive_y)


                # Report Loss and Accuracy
                d_loss_total[0] += d_loss[0]
                d_loss_total[1] = ((d_loss_total[1]*batch_size*batch_num) + d_loss[1]*batch_size)/((batch_num+1)*batch_size)
                a_loss_total[0] += a_loss[0]
                a_loss_total[1] = ((a_loss_total[1]*batch_size*batch_num) + a_loss[1]*batch_size)/((batch_num+1)*batch_size)
                pbar.set_description("D_acc: {:.3f},A_acc: {:.3f}".format(d_loss_total[1], a_loss_total[1]))

            # Epoch ended
            # Print epoch statistics
            endtime = datetime.now()
            print("    epoch time: {}".format(endtime-starttime))

            print("    d_loss: {}".format(d_loss_total))
            print("    a_loss: {}".format(a_loss_total))

            # Get example output from generator
            displayed_samples = self.generator.predict(np.random.uniform(-1.0, 1.0, (5, 100)))

            row = [epoch, d_loss_total[0], d_loss_total[1], a_loss_total[0], a_loss_total[1]]
            with open(os.path.join(run_directory, 'log.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)


            if epoch%img_interval == 0:
                self.adversarial.save(os.path.join(model_dir, 'adversarial_checkpoint_acc{}_e{}.h5'.format(a_loss_total[1], epoch)))                
                np.save('generated_samples_e{}.npy'.format(epoch), displayed_samples)
            if a_loss_total[0] >= last_a_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                last_a_loss = a_loss_total[0]

            if patience_counter >= patience:
                print("Adversarial loss did not improve for {} epochs. Stopping early at epoch {}...".format(patience, epoch))
                break

        self.adversarial.save(os.path.join(model_dir, 'adversarial_final_acc{}.h5'.format(a_loss_total[1])))                
        np.save('generated_samples_final.npy', displayed_samples)
        #  wio.write(os.path.join(run_directory, "reconstrution_final.wav"), 44100, reconstruction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file')
    #  parser.add_argument('phase_file')
    args = parser.parse_args()
    waveletGAN = FSDD_SpectroGAN(args.dataset_file) #, args.phase_file)
    timer = ElapsedTimer()
    waveletGAN.train_GAN(num_epochs=50, batch_size=8, img_interval=1)
    timer.elapsed_time()
