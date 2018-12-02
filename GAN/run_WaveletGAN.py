import numpy as np
from datetime import datetime
import time
import os
import sys
import csv
import pywt
import librosa
#  import scipy.io.wavfile as wio
from tqdm import tqdm
from WaveletGAN import WaveletGAN

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

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

class FSDD_WaveletGAN(object):
    def __init__(self):
        self.img_rows = 591
        self.img_cols = 944

        self.datagen = ImageDataGenerator()

        # self.x_train = input_data.read_data_sets("mnist",\
        #     one_hot=True).train.images
        # self.x_train = self.x_train.reshape(-1, self.img_rows,\
        #     self.img_cols, 1).astype(np.float32)

        xd = np.load('bird_wavelets.npy')
        xd = xd[0]
        if xd.shape[1] % 2 != 0:
            xd = xd[:,:-1,:]
        self.WaveletGAN = WaveletGAN(img_rows=xd.shape[0], img_cols=xd.shape[1])
        self.discriminator =  self.WaveletGAN.discriminator_model()
        self.adversarial = self.WaveletGAN.adversarial_model()
        self.generator = self.WaveletGAN.generator()

    def train_GAN(self, num_epochs=100, batch_size=32, img_interval = 10, patience=5):
        
        datestr = "{:%m%d%y_%H%M%S}".format(datetime.now())
        run_directory = 'bird_wavelet_runs/{}/'.format(datestr)
        os.makedirs(run_directory, exist_ok=True)
        model_dir = os.path.join(run_directory, 'discriminator_models')
        os.makedirs(model_dir, exist_ok=True)

        x_data = np.load('bird_wavelets.npy')
        print("============================================\r\n======================================================\r\n")
        print("x_data: {}".format(x_data.shape))
        y_labels = np.ones((len(x_data)))
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

            train_gen = self.datagen.flow(x_data, y_labels,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          seed=0
                                         )

            batch_num = 0

            d_loss_total = [0.0, 0.0]
            a_loss_total = [0.0, 0.0]
            starttime = datetime.now()

            # Get Batch
            for x_batch,y_batch in tqdm( train_gen ):
                _,_,N,_ = x_batch.shape
                if N % 2 != 0:
                    x_batch = x_batch[:,:,:-1,:]
                # Augment real data with fake data
                noise_vectors = np.random.uniform(-1.0, 1.0, size=(len(x_batch), 100))
                fake_ims = self.generator.predict(noise_vectors)
                #  fake_ims = np.random.random(x_batch.shape)
                if len(fake_ims) >= 4:
                    displayed_samples = fake_ims
                x_batch = np.concatenate((x_batch, fake_ims))
                y_batch = np.hstack([y_batch, np.zeros(y_batch.shape)])

                # Run Discriminator
                d_loss = self.discriminator.train_on_batch(x_batch, y_batch)
                d_loss_total[0] += d_loss[0]
                d_loss_total[1] = ((d_loss_total[1]*batch_size*batch_num) + d_loss[1]*batch_size)/((batch_num+1)*batch_size)

                # Run Adversarial
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                noise_labels = np.ones(len(noise))
                a_loss = self.adversarial.train_on_batch(noise, noise_labels)
                a_loss_total[0] += a_loss[0]
                a_loss_total[1] = ((a_loss_total[1]*batch_size*batch_num) + a_loss[1]*batch_size)/((batch_num+1)*batch_size)

                #  # Run Adversarial again
                #  noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                #  noise_labels = np.ones(len(noise))
                #  a_loss = self.adversarial.train_on_batch(noise, noise_labels)
                #  a_loss_total[0] += a_loss[0]
                #  a_loss_total[1] = ((a_loss_total[1]*batch_size*batch_num) + a_loss[1]*batch_size)/((batch_num+1)*batch_size)

                batch_num += 1
                if batch_num*batch_size > num_samples:
                    break

            endtime = datetime.now()
            print("    epoch time: {}".format(endtime-starttime))

            print("    d_loss: {}".format(d_loss_total))
            print("    a_loss: {}".format(a_loss_total))


            row = [epoch, d_loss_total[0], d_loss_total[1], a_loss_total[0], a_loss_total[1]]
            with open(os.path.join(run_directory, 'log.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)


            if epoch%img_interval == 0:
                self.adversarial.save(os.path.join(model_dir, 'adversarial_checkpoint_acc{}_e{}.h5'.format(a_loss_total[1], epoch)))                
                wavelet = displayed_samples[0]
                cA, cD = wavelet[0].reshape(-1), wavelet[1].reshape(-1)
                reconstruction = pywt.idwt(cA, cD, 'db2')
                librosa.output.write_wav(os.path.join(run_directory, "reconstruction_e{}.wav".format(epoch)), reconstruction, 44100, norm=True)
                #  wio.write(os.path.join(run_directory, "reconstrution_e{}.wav".format(epoch)), 44100, reconstruction)
            if a_loss_total[0] >= last_a_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                last_a_loss = a_loss_total[0]

            if patience_counter >= patience:
                print("Adversarial loss did not improve for {} epochs. Stopping early at epoch {}...".format(patience, epoch))
                break

        self.adversarial.save(os.path.join(model_dir, 'adversarial_final_acc{}.h5'.format(a_loss_total[1])))                
        wavelet = displayed_samples[0]
        cA, cD = wavelet[0].reshape(-1), wavelet[1].reshape(-1)
        reconstruction = pywt.idwt(cA, cD, 'db2')
        librosa.output.write_wav(os.path.join(run_directory, "reconstruction_final.wav"), reconstruction, 44100, norm=True)
        #  wio.write(os.path.join(run_directory, "reconstrution_final.wav"), 44100, reconstruction)

if __name__ == '__main__':
    waveletGAN = FSDD_WaveletGAN()
    timer = ElapsedTimer()
    waveletGAN.train_GAN(num_epochs=50, batch_size=8, img_interval=1)
    timer.elapsed_time()
