from config import img_size
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from data import prepare_dataset, preprocess_image
import matplotlib.pyplot as plt

from loss import vae_loss
from model import encoder, decoder, latent_sample
from config import save_every
import tensorflow as tf
import numpy as np
import time
import os

from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

file_count = sum(len(files) for _, _, files in os.walk(r'data'))


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

    def train_step(self, batch):

        images = np.stack(batch)
        with tf.GradientTape() as tape:


            latent_mu, latent_logvar = self.encoder(images, training=False)
            latent = latent_sample(latent_mu, latent_logvar)
            x_recon = self.decoder(latent, training=False)
            loss = self.loss_fn(x_recon, images, latent_mu, latent_logvar)

        gradients = tape.gradient(loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_weights))
        return loss

    def train(self):
        start_time = time.time()
        dataset = prepare_dataset()

        # self.normalizer.adapt(np.array([next(dataset) for _ in range(250)]))  # 250

        epoch_start = 0
        print('starting at:', epoch_start)

        # bar = tf.keras.utils.Progbar(file_count * dataset_repetitions / batch_size - 1)
        for epoch, batch in enumerate(dataset):
            self.train_step(batch)
            # if (epoch + epoch_start) % save_every == 0:
            #     with open("logs.txt", "a") as file_object:
            #         file_object.write(
            #             "\n" + f'epoch: {epoch + epoch_start} time: {time.time() - start_time} loss: {loss} ')
            # self.plot_images(epoch + epoch_start)


g = VAE()
g.train()


class Diffusion:

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, img_size, img_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(num_rows * num_cols, plot_diffusion_steps)

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"output_images/generated_plot_epoch-{epoch}.png")
