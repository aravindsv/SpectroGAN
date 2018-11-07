import numpy as np
from datetime import datetime
import time
import os
import sys
import csv
from tqdm import tqdm
from DCGAN import DCGAN

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

train_dir = './Datagen/Train'
test_dir = './Datagen/Test'

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

class FSDD_DCGAN(object):
    def __init__(self):
        self.img_rows = 591
        self.img_cols = 944
        self.channel = 3

        self.datagen = ImageDataGenerator()

        # self.x_train = input_data.read_data_sets("mnist",\
        #     one_hot=True).train.images
        # self.x_train = self.x_train.reshape(-1, self.img_rows,\
        #     self.img_cols, 1).astype(np.float32)

        # self.DCGAN = DCGAN(img_rows=self.img_rows, img_cols=self.img_cols, channel=self.channel)
        self.DCGAN = DCGAN(img_rows=256, img_cols=256, channel=self.channel)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train_GAN(self, num_epochs=100, batch_size=32, img_interval = 10, patience=5):
        
        datestr = "{:%m%d%y_%H%M%S}".format(datetime.now())
        run_directory = 'bird_runs/{}/'.format(datestr)
        os.makedirs(run_directory, exist_ok=True)
        model_dir = os.path.join(run_directory, 'discriminator_models')
        os.makedirs(model_dir, exist_ok=True)

        x_data = np.load('bird_imgs.npy')
        y_labels = np.ones((len(x_data)))
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
            for x_batch,y_batch in train_gen:
                # Augment real data with fake data
                noise_vectors = np.random.uniform(-1.0, 1.0, size=(len(x_batch), 100))
                fake_ims = self.generator.predict(noise_vectors)
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
                plt.figure()
                plt.title("Generated Images at Epoch {}, Accuracy: {}".format(epoch, a_loss_total[1]))
                plt.subplot(2,2,1)
                plt.axis('off')
                plt.imshow(displayed_samples[0])
                plt.subplot(2,2,2)
                plt.axis('off')
                plt.imshow(displayed_samples[1])
                plt.subplot(2,2,3)
                plt.axis('off')
                plt.imshow(displayed_samples[2])
                plt.subplot(2,2,4)
                plt.axis('off')
                plt.imshow(displayed_samples[3])
                plt.savefig(os.path.join(run_directory, 'images_e{}'.format(epoch)))

            if a_loss_total[0] >= last_a_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                last_a_loss = a_loss_total[0]

            if patience_counter >= patience:
                print("Adversarial loss did not improve for {} epochs. Stopping early at epoch {}...".format(patience, epoch))
                break

        self.adversarial.save(os.path.join(model_dir, 'adversarial_final_acc{}.h5'.format(a_loss_total[1])))                
        plt.figure()
        plt.title("Final Generated Images, Accuracy: {}".format(a_loss_total[1]))
        plt.subplot(2,2,1)
        plt.axis('off')
        plt.imshow(displayed_samples[0])
        plt.subplot(2,2,2)
        plt.axis('off')
        plt.imshow(displayed_samples[1])
        plt.subplot(2,2,3)
        plt.axis('off')
        plt.imshow(displayed_samples[2])
        plt.subplot(2,2,4)
        plt.axis('off')
        plt.imshow(displayed_samples[3])
        plt.savefig(os.path.join(run_directory, 'images_final'))


    # def train(self, train_steps=2000, batch_size=32, save_interval=0):
    #     noise_input = None
    #     train_gen = self.datagen.flow_from_directory(train_dir,
    #                                                  target_size=(self.img_rows, self.img_cols),
    #                                                  color_mode='rgb',
    #                                                  class_mode='sparse',
    #                                                  batch_size=batch_size,
    #                                                  shuffle=True,
    #                                                  seed=0)

    #     if save_interval>0:
    #         noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    #     datestr = "{:%m%d%y_%H%M%S}".format(datetime.now())
    #     for i in range(train_steps):
    #         print("Step {}/{}".format(i, train_steps))

    #         # images_train = self.x_train[np.random.randint(0,
    #         #     self.x_train.shape[0], size=batch_size), :, :, :]
    #         # noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    #         # images_fake = self.generator.predict(noise)
            
    #         x,y = train_gen.__next__()
    #         # x = np.concatenate((images_train, images_fake))
    #         # y = np.ones([2*batch_size, 1])
    #         # y[batch_size:, :] = 0
    #         d_loss = self.discriminator.train_on_batch(x, y)

    #         #  y = np.ones([batch_size, 1])
    #         #  noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    #         #  a_loss = self.adversarial.train_on_batch(noise, y)
    #         log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
    #         #  log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
    #         print(log_mesg)
    #         #  if save_interval>0:
    #         #      if (i+1)%save_interval==0:
    #         #          self.plot_images(save2file=True, samples=noise_input.shape[0],\
    #         #              noise=noise_input, step=(i+1))

    #         #      os.makedirs('models/adversarial_{}'.format(datestr))
    #         #      self.adversarial.save('models/adversarial_{}/step_{}_acc_{}.h5'.format(datestr, i, a_loss[1]))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'fsdd.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "fsdd_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    spectroGAN = FSDD_DCGAN()
    timer = ElapsedTimer()
    spectroGAN.train_GAN(num_epochs=50, batch_size=8)
    timer.elapsed_time()
    # spectroGAN.plot_images(fake=True)
    # spectroGAN.plot_images(fake=False, save2file=True)
