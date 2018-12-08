from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import UpSampling1D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.layers.merge import _Merge
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import numpy as np
from functools import partial

GRADIENT_PENALTY_WEIGHT = 10
BATCH_SIZE = 8

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

class EiGAN(object):
    def __init__(self, num_components=500):
        self.num_components = num_components

        self.D = None
        self.G = None
        self.AM = None
        self.DM = None


    def discriminator(self):
        if self.D:
            return self.D

        self.D = Sequential()

        # Dense Segment 1
        self.D.add(Dense(1024))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(0.25))

        # Dense Segment 2
        self.D.add(Dense(512))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(0.25))

        # Dense Segment 3
        self.D.add(Dense(256))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(0.25))

        # Dense Segment 4
        self.D.add(Dense(256))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(0.25))

        # Final layer
        self.D.add(Dense(1))
        self.D.activation(Activation('linear'))
        print("Discriminator Summary:")
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G

        self.G = Sequential()

        starting_dim = self.num_components // 4

        self.G.add(Dense(starting_dim, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape(starting_dim,))
        self.G.add(Dropout(0.25))

        self.G.add(Upsampling1D())
        self.G.add(Dense(starting_dim*2))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Upsampling1D())
        self.G.add(Dense(starting_dim*4))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('sigmoid'))

        print("Generator Summary:")
        self.G.summary()
        return self.G

    
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
        loss_fn = wasserstein_loss
        D = self.discriminator()
        G = self.generator()
        for layer in D.layers:
            layer.trainable = True
        for layer in G.layers:
            layer.trainable = False
        D.trainable = True
        G.trainable = False
        real_samples = Input(shape=(self.num_components))
        generator_input_for_discriminator = Input(shape=(100,))
        generated_samples_for_discriminator = G(generator_input_for_discriminator)
        discriminator_output_from_generator = D(generated_samples_for_discriminator)
        discriminator_output_from_real_samples = D(real_samples)
        print("real_samples: {}".format(real_samples.shape))
        print("generated_samples_for_discriminator: {}".format(generated_samples_for_discriminator.shape))

        averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
        averaged_samples_out = D(averaged_samples)

        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.DM = Model(inputs=[real_samples, generator_input_for_discriminator],
                        outputs=[discriminator_output_from_real_samples,
                                 discriminator_output_from_generator,
                                 averaged_samples_out])
        self.DM.compile(optimizer=optimizer, loss=[loss_fn, loss_fn, partial_gp_loss])

        return self.DM


    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
        loss_fn = wasserstein_loss
        D = self.discriminator()
        G = self.generator()
        for layer in D.layers:
            layer.trainable = False
        for layer in G.layers:
            layer.trainable = True
        D.trainable = False
        G.trainable = True

        generator_input = Input(shape=(100,))
        generator_layers = G(generator_input)
        discriminator_layers_for_generator = D(generator_layers)
        self.AM = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
        self.AM.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        return self.AM
