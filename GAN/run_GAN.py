import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pywt
import librosa
import scipy.io.wavfile as wio

import torchvision.datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import tensorboardX

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels=1):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128*self.init_size**2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels=1, w_loss=False):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3,2,1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        ds_size = img_size // 2**4
        adv_layers = [nn.Linear(128*ds_size**2, 1)]
        if not w_loss:
            adv_layers.append(nn.Sigmoid())
        self.adv_layer = nn.Sequential(*adv_layers)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', help='Directory with images to train off of')
    parser.add_argument('datatype', choices=['wav', 'spec', 'wavelet'], default='spec', help='The type of data to train on')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
    parser.add_argument('--w_loss', action='store_true', help='Use Wasserstein Loss instead of BCE loss')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='adam: learning rate for discriminator')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='adam: learning rate for generator')
    parser.add_argument('--discriminator_cycles', type=int, default=1, help='Number of times to run the discriminator')
    parser.add_argument('--generator_cycles', type=int, default=1, help='Number of times to run the generator')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=20, help='interval between image sampling')
    args = parser.parse_args()
    return args

class wavfile2Array(object):
    """ Converts wavfile to numpy array for general use """
    def __init__(self, sr=None):
        self.sr = sr

    def __call__(self, wavfile):
        wav, fs = librosa.core.load(wavfile, sr=self.sr)
        return wav

class array2Spectrogram(object):
    """Convert wavfile to a numpy array spectrogram"""
    def __init__(self, nfft=2048, hop_length=256, window_sz=None):
        self.nfft = nfft
        self.hop_length = hop_length
        self.window = np.hanning(nfft) if window_sz is None else np.hanning(window_sz)

    def __call__(self, wav):
        spectrogram = librosa.core.stft(wav, 
                                        n_fft=self.nfft,
                                        hop_length=self.hop_length,
                                        window=self.window)
        M = np.abs(spectrogram)
        M = M.reshape(M.shape[0], M.shape[1], 1)
        return M


class array2Wavelet(object):
    """ Convert wavfile to a wavelet array """
    def __init__(self, square=None):
        self.square = square

    def __call__(self, wav):
        cA, cD = pywt.dwt(wav, 'db2')
        desired_len = int((len(cA) // 4) * 4)
        cA, cD = cA[:desired_len], cD[:desired_len]

        waveletc = np.stack([cA, cD])
        if self.square is not None:
            waveletc = waveletc.reshape(self.square)

        return waveletc



if __name__ == "__main__":
    args = getArgs()
    cuda = True if torch.cuda.is_available() else False

    # Make important directories
    foldername = "{:%b%d_%H%M%S}_{}_lrd{}_lrg{}_e{}_bs{}".format(datetime.now(), args.dataset_folder.split('/')[-1], args.lr_d, args.lr_g, args.n_epochs, args.batch_size)
    rundir = 'runs/{}'.format(foldername)
    resultdir = os.path.join(rundir, 'results')

    os.makedirs(resultdir, exist_ok=True)

    with open(os.path.join(rundir, 'commandline_args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[:]))

    tbwriter = tensorboardX.SummaryWriter(rundir)

    # Create Loss Function
    adversarial_loss = nn.BCELoss()
    # TODO: Implement w_loss

    # create generator and discriminator
    generator = Generator(args.img_size, args.latent_dim, channels=args.channels)
    discriminator = Discriminator(args.img_size, channels=args.channels)
    # TODO: move generator and discriminator into separate files
    # TODO: write a separate generator and discriminator for 1D audio

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Create transforms
    transformations = {
        'wav': []
        'spec': [array2Spectrogram(), transforms.Grayscale(), transforms.Resize(args.img_size)],
        'wavelet': [array2Wavelet()]
    }[args.datatype]

    transformations.extend([
        transforms.ToTensor()
    ])

    transform = transforms.Compose(transformations)

    # Configure data loader
    dataset = torchvision.datasets.DatasetFolder(root=args.dataset_folder,
                                                 loader=wavfile2Array,
                                                 transform=transform,
                                                )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(args.n_epochs):
        start_time = time.time()
        num_samples = 0
        g_acc = 0.0
        d_acc = 0.0
        g_loss_total = 0.0
        d_loss_total = 0.0

        for batch_num, (imgs, _) in enumerate(dataloader):
            # Adversarial setup
            real_labels = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake_labels = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            real_imgs = Variable(imgs.type(Tensor))

            cur_batch_size = imgs.shape[0]

            # First train generator
            # Don't allow discriminator to be trained during this part
            for p in discriminator.parameters():
                p.requires_grad = False
            for p in generator.parameters():
                p.requires_grad = True

            for i in range(args.generator_cycles):
                optimizer_G.zero_grad()

                noisevec = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

                fake_imgs = generator(noisevec)

                predictions_for_fake_imgs = discriminator(fake_imgs)

                g_loss = adversarial_loss(predictions_for_fake_imgs, real_labels)
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

            g_num_correct = predictions_for_fake_imgs.round().sum()
            g_acc = ((g_acc * num_samples) + g_num_correct) / (num_samples + cur_batch_size)
            g_loss_total += g_loss.item()


            # Then train discriminator
            # Don't allow generator to be trained during this part
            for p in discriminator.parameters():
                p.requires_grad = True
            for p in generator.parameters():
                p.requires_grad = False

            for i in range(args.discriminator_cycles):
                optimizer_D.zero_grad()

                predictions_for_real_imgs = discriminator(real_imgs)
                predictions_for_noise_imgs = discriminator(fake_imgs)


                real_loss = adversarial_loss(predictions_for_real_imgs, real_labels)
                fake_loss = adversarial_loss(predictions_for_noise_imgs, fake_labels)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

            d_num_correct = predictions_for_real_imgs.round().sum()
            d_num_correct += (len(predictions_for_noise_imgs) - (predictions_for_noise_imgs.round().sum()))
            d_acc = ((d_acc * num_samples) + d_num_correct) / (2*(num_samples+cur_batch_size))
            d_loss_total += d_loss.item()

            print("    Batch {}/{}, D_loss = {:.3f}, G_loss = {:.3f}, D_acc = {:.3f}, G_acc = {:.3f}".format(batch_num+1, len(dataloader), d_loss_total, g_loss_total, d_acc, g_acc), end='\r')
            num_samples += cur_batch_size

        print("")
        tbwriter.add_scalars('Loss', {'Generator_Loss': g_loss_total,
                                      'Discriminator_Loss': d_loss_total}
                                   , epoch)
        tbwriter.add_scalars('Acc', {'Generator_Acc': g_acc,
                                      'Discriminator_Acc': d_acc}
                                   , epoch)
        print("    D_loss = {:.3f}, G_loss = {:.3f}, D_acc = {:.3f}, G_acc = {:.3f}".format(d_loss_total, g_loss_total, d_acc, g_acc))

        if epoch % args.sample_interval == 0:
            np.save(os.path.join(resultdir, 'results_e{}.npy'.format(epoch)), fake_imgs)