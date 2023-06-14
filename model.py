from keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose, Reshape
from tensorflow import keras, exp, random
from config import img_size, capacity


def encoder():
    images = keras.Input(shape=(img_size, img_size, 3))

    x = Conv2D(capacity, kernel_size=4, strides=2, padding="same", activation='relu', name='kok')(images)  # 256X256
    x = Conv2D(2 * capacity, kernel_size=4, strides=2, padding="same", activation='relu', name='vvvvvvv')(x)  # 128X128
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
    encoded = keras.Input(shape=(1000,))
    x = Dense(8192)(encoded)
    x = Reshape((4, 4, 64 * capacity))(x)

    x = Conv2DTranspose(32 * capacity, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(16 * capacity, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(8 * capacity, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(4 * capacity, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(2 * capacity, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)

    return keras.Model(encoded, x, name="decoder")


def latent_sample(mu, logvar, training=False):
    if training:
        std = exp(logvar * 0.5)
        eps = random.normal(std.shape)
        return eps * std + mu
    else:
        return mu
