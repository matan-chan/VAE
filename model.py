from keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose, Reshape
from config import img_size, capacity
from tensorflow import keras, Tensor
from typing import Tuple, Union
from numpy import ndarray
import tensorflow as tf
import numpy as np


def log_normal_pdf(sample: Tensor, mean: Union[float, Tensor], logvar: Union[float, Tensor], raxis: int = 1) -> Tensor:
    log2pi = tf.math.log(2. * np.pi)
    log_normal_pdf = tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)
    return log_normal_pdf


def encoder() -> keras.Model:
    images = keras.Input(shape=(img_size, img_size, 3))

    x = Conv2D(capacity, kernel_size=3, strides=2, padding="same", activation='relu')(images)  # 128X128
    x = Conv2D(2 * capacity, kernel_size=3, strides=2, padding="same", activation='relu')(x)  # 64X64
    x = Conv2D(4 * capacity, kernel_size=3, strides=2, padding="same", activation='relu')(x)  # 32X32

    x = Flatten()(x)
    x = Dense(6000)(x)

    return keras.Model(images, x, name="encoder")


def decoder() -> keras.Model:
    encoded = keras.Input(shape=(3000,))
    x = Dense(32768)(encoded)
    x = Reshape((32, 32, 4 * capacity))(x)

    x = Conv2DTranspose(4 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(2 * capacity, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(3, kernel_size=3, strides=2, padding='same')(x)

    return keras.Model(encoded, x, name="decoder")


class VAE(keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, load_from_file: bool = False):
        super(VAE, self).__init__()
        if load_from_file:
            self.encoder = tf.keras.models.load_model('models/encoder_model.h5')
            self.decoder = tf.keras.models.load_model('models/decoder_model.h5')
        else:
            self.encoder = encoder()
            self.decoder = decoder()

    @tf.function
    def sample(self, eps=None) -> Tensor:
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x: ndarray) -> Tuple[Tensor, Tensor]:
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var

    @staticmethod
    def reparameterize(mean: Tensor, log_var: Tensor) -> Tensor:
        eps = tf.random.normal(shape=mean.shape)
        z = eps * tf.exp(log_var * .5) + mean
        return z

    def decode(self, z: Tensor, apply_sigmoid: bool = False) -> Tensor:
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compress(self, images: ndarray) -> Tensor:
        mean, log_var = self.encode(images)
        z = self.reparameterize(mean, log_var)
        return z

    def expand(self, z: Tensor) -> ndarray:
        predicts = self.decode(z)
        predicts = np.array(tf.sigmoid(predicts))
        image = (predicts * 255).astype(np.uint8)
        return image
