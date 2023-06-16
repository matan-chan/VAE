import cv2

from config import img_size, dataset_repetitions, batch_size
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from data import prepare_dataset, preprocess_image, get_dataframe
import matplotlib.pyplot as plt
from PIL import Image
from loss import vae_loss
from model import encoder, decoder, latent_sample
from config import save_every
import tensorflow as tf
import numpy as np
import time
import os

from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

file_count = len(get_dataframe())


class VAE():
    def __init__(self):
        self.encoder = encoder()
        self.decoder = decoder()
        images = tf.keras.Input(shape=(img_size, img_size, 3))
        valid = self.decoder(self.encoder(images))
        self.vae = tf.keras.Model(images, valid)
        self.optimizer = Adam(learning_rate=1e-4)

    @staticmethod
    def loss_fn(recon_x, x, mu, logvar):
        return vae_loss(recon_x, x, mu, logvar)

    def train_step(self, images):
        with tf.GradientTape() as tape:
            latent_mu, latent_logvar = self.encoder(images, training=False)
            latent = latent_sample(latent_mu, latent_logvar)
            x_recon = self.decoder(latent, training=False)
            loss = self.loss_fn(x_recon, images, latent_mu, latent_logvar)

        gradients = tape.gradient(loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_weights))
        return loss, x_recon

    # def plot_images(self,butch, epoch=None, num_rows=2, num_cols=6):
    #     # plot random generated images for visual evaluation of generation quality
    #     generated_images = self.encoder(butch)
    #     generated_images = self.decoder(butch)
    #
    #     plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
    #     for row in range(num_rows):
    #         for col in range(num_cols):
    #             index = row * num_cols + col
    #             plt.subplot(num_rows, num_cols, index + 1)
    #             plt.imshow(generated_images[index])
    #             plt.axis("off")
    #     plt.tight_layout()
    #     plt.savefig(f"output_images/generated_plot_epoch-{epoch}.png")

    def convert_to_h5(self):
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder,
                                         vae=self.vae)
        manager = tf.train.CheckpointManager(checkpoint, directory='models/', max_to_keep=3)
        self.encoder.save('models/encoder_model.h5')
        self.decoder.save('models/decoder_model.h5')
        self.vae.save('models/vae_model.h5')

    def train(self):
        dataset = prepare_dataset(True)

        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder,
                                         vae=self.vae)
        manager = tf.train.CheckpointManager(checkpoint, directory='models/', max_to_keep=3)

        epoch_start = int(
            manager.latest_checkpoint.split(sep='ckpt-')[
                -1]) * save_every if manager.latest_checkpoint else 1
        print('starting at:', epoch_start)
        checkpoint.restore(manager.latest_checkpoint)
        bar = tf.keras.utils.Progbar(file_count * dataset_repetitions / batch_size - 1)
        for epoch, batch in enumerate(dataset):
            loss, x_recon = self.train_step(batch)
            loss = np.array(loss)

            bar.update(epoch + epoch_start, values=[("loss", loss)])
            if (epoch + epoch_start) % save_every == 0:
                # self.plot_images(epoch + epoch_start)

                im_rgb = cv2.cvtColor(x_recon, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f'output_images/{epoch}.jpg', im_rgb)

                manager.save()


g = VAE()
g.train()
