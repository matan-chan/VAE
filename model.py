import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose, Reshape, Lambda
from tensorflow import keras, exp, random
from config import img_size, capacity
import tensorflow as tf


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def encoder():
    images = keras.Input(shape=(img_size, img_size, 3))

    # x = Lambda(normalize)(images)

    x = Conv2D(capacity, kernel_size=5, strides=2, padding="same", activation='relu')(images)  # 128X128
    x = Conv2D(capacity, kernel_size=5, strides=1, padding="same", activation='relu')(x)  # 128X128
    x = Conv2D(capacity, kernel_size=5, strides=1, padding="same", activation='relu')(x)  # 128X128
    x = Conv2D(2 * capacity, kernel_size=5, strides=2, padding="same", activation='relu')(x)  # 64X64
    x = Conv2D(2 * capacity, kernel_size=5, strides=1, padding="same", activation='relu')(x)  # 64X64
    x = Conv2D(2 * capacity, kernel_size=5, strides=1, padding="same", activation='relu')(x)  # 64X64
    x = Conv2D(4 * capacity, kernel_size=3, strides=2, padding="same", activation='relu')(x)  # 32X32
    x = Conv2D(4 * capacity, kernel_size=3, strides=1, padding="same", activation='relu')(x)  # 32X32
    x = Conv2D(8 * capacity, kernel_size=3, strides=2, padding="same", activation='relu')(x)  # 16X16
    x = Conv2D(16 * capacity, kernel_size=3, strides=2, padding="same", activation='relu')(x)  # 8X8


    x = Flatten()(x)
    x = Dense(6000)(x)

    return keras.Model(images, x, name="encoder")


def decoder():
    encoded = keras.Input(shape=(3000,))
    x = Dense(8192)(encoded)
    x = Reshape((8, 8, 16 * capacity))(x)


    x = Conv2DTranspose(16 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(8 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(4 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(2 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid')(x)

    return keras.Model(encoded, x, name="decoder")


class VAE(keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, ):
        super(VAE, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.encoder.summary()
        self.decoder.summary()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits




# import numpy as np
# from keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose, Reshape, Lambda
# from tensorflow import keras, exp, random
# from config import img_size, capacity
# import tensorflow as tf
#
#
# def log_normal_pdf(sample, mean, logvar, raxis=1):
#     log2pi = tf.math.log(2. * np.pi)
#     return tf.reduce_sum(
#         -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#         axis=raxis)
#
#
# def encoder():
#     images = keras.Input(shape=(img_size, img_size, 3))
#
#     # x = Lambda(normalize)(images)
#
#     x = Conv2D(capacity, kernel_size=3, strides=2, padding="same", activation='relu', name='kok')(images)  # 128X128
#     x = Conv2D(2 * capacity, kernel_size=3, strides=2, padding="same", activation='relu', name='vvvvvvv')(x)  # 64X64
#     x = Conv2D(4 * capacity, kernel_size=3, strides=2, padding="same", activation='relu')(x)  # 32X32
#     x = Conv2D(8 * capacity, kernel_size=3, strides=2, padding="same", activation='relu')(x)  # 16X16
#     x = Conv2D(16 * capacity, kernel_size=3, strides=2, padding="same", activation='relu')(x)  # 8X8
#
#
#     x = Flatten()(x)
#     x = Dense(6000)(x)
#
#     return keras.Model(images, x, name="encoder")
#
#
# def decoder():
#     encoded = keras.Input(shape=(3000,))
#     x = Dense(8192)(encoded)
#     x = Reshape((8, 8, 16 * capacity))(x)
#
#
#     x = Conv2DTranspose(16 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = Conv2DTranspose(8 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = Conv2DTranspose(4 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = Conv2DTranspose(2 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid')(x)
#
#     return keras.Model(encoded, x, name="decoder")
#
#
# class VAE(keras.Model):
#     """Convolutional variational autoencoder."""
#
#     def __init__(self, ):
#         super(VAE, self).__init__()
#         self.encoder = encoder()
#         self.decoder = decoder()
#         self.encoder.summary()
#         self.decoder.summary()
#
#     @tf.function
#     def sample(self, eps=None):
#         if eps is None:
#             eps = tf.random.normal(shape=(100, self.latent_dim))
#         return self.decode(eps, apply_sigmoid=True)
#
#     def encode(self, x):
#         mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
#         return mean, logvar
#
#     def reparameterize(self, mean, logvar):
#         eps = tf.random.normal(shape=mean.shape)
#         return eps * tf.exp(logvar * .5) + mean
#
#     def decode(self, z, apply_sigmoid=False):
#         logits = self.decoder(z)
#         if apply_sigmoid:
#             probs = tf.sigmoid(logits)
#             return probs
#         return logits



