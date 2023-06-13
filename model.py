from keras.backend import softmax, transpose, reshape, shape
from keras.layers import Conv2D, BatchNormalization, Add, AveragePooling2D, UpSampling2D, Concatenate, Lambda, Dense, \
    Flatten, Conv2DTranspose, Reshape
from tensorflow.python.layers.base import Layer

from config import img_size, capacity
from tensorflow import keras, matmul, exp
import tensorflow as tf
import math


def encoder():
    images = keras.Input(shape=(img_size, img_size, 3))

    x = Conv2D(capacity, kernel_size=4, strides=2, padding="same", activation='relu')(images)  # 256X256
    x = Conv2D(2 * capacity, kernel_size=4, strides=2, padding="same", activation='relu')(x)  # 128X128
    x = Conv2D(4 * capacity, kernel_size=4, strides=2, padding="same", activation='relu')(x)  # 64X64
    x = Conv2D(8 * capacity, kernel_size=4, strides=2, padding="same", activation='relu')(x)  # 32X32
    x = Conv2D(16 * capacity, kernel_size=4, strides=2, padding="same", activation='relu')(x)  # 16X16
    x = Conv2D(32 * capacity, kernel_size=4, strides=2, padding="same", activation='relu')(x)  # 8X8
    x = Conv2D(64 * capacity, kernel_size=4, strides=2, padding="same", activation='relu')(x)  # 4X4 = 8192

    x = Flatten()(x)
    fc_mu = Dense(1000)(x)
    fc_logvar = Dense(1000)(x)
    return keras.Model(images, [fc_mu, fc_logvar], name="encoder")


def decoder():
    encoded = keras.Input(shape=(1, 1000))
    x = Dense(8192)(encoded)
    x = Reshape((64 * capacity, 4, 4))(x)

    x = Conv2DTranspose(32 * capacity, kernel_size=4, stride=2, activation='relu')(x)
    x = Conv2DTranspose(16 * capacity, kernel_size=4, stride=2, activation='relu')(x)
    x = Conv2DTranspose(8 * capacity, kernel_size=4, stride=2, activation='relu')(x)
    x = Conv2DTranspose(4 * capacity, kernel_size=4, stride=2, activation='relu')(x)
    x = Conv2DTranspose(3, kernel_size=4, stride=2, activation='sigmoid')(x)

    return keras.Model(encoded, x, name="encoder")


def latent_sample(mu, logvar, training=False):
    if training:
        std = exp(logvar * 0.5)
        eps = tf.random.normal(std.shape)
        return eps * std + mu
    else:
        return mu


def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence
