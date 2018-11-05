import numpy as np
import time
import os
from DCGAN import DCGAN

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

train_dir = './free-spoken-digit-dataset-master/Datagen/Train'
test_dir = './free-spoken-digit-dataset-master/Datagen/Test'

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

        self.DCGAN = DCGAN(img_rows=self.img_rows, img_cols=self.img_cols, channel=self.channel)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=32, save_interval=0):
        noise_input = None
        train_gen = self.datagen.flow_from_directory(train_dir,
                                                     target_size=(self.img_rows, self.img_cols),
                                                     color_mode='rgb',
                                                     class_mode='sparse',
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     seed=0)

        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):

            # images_train = self.x_train[np.random.randint(0,
            #     self.x_train.shape[0], size=batch_size), :, :, :]
            # noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            # images_fake = self.generator.predict(noise)
            
            x,y = train_gen.__next__()
            # x = np.concatenate((images_train, images_fake))
            # y = np.ones([2*batch_size, 1])
            # y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'fsdd.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
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
    spectroGAN.train(train_steps=10000, batch_size=256, save_interval=500)
    timer.elapsed_time()
    spectroGAN.plot_images(fake=True)
    spectroGAN.plot_images(fake=False, save2file=True)