import cv2

from config import img_size, dataset_repetitions, batch_size
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from data import prepare_dataset, preprocess_image, get_dataframe
import matplotlib.pyplot as plt
from PIL import Image
from loss import vae_loss
from model import encoder, decoder, log_normal_pdf, VAE
from config import save_every
import tensorflow as tf
import numpy as np
import time
import cv2
import os

from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

file_count = sum(len(files) for _, _, files in os.walk(r'data/images'))


class VAE_interface():
    def __init__(self):
        self.model = VAE()
        self.optimizer = Adam(learning_rate=1e-4)

    def compute_loss(self, x):
        mean, logvar = self.model.encode(x)
        z = self.model.reparameterize(mean, logvar)
        x_logit = self.model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x), x_logit

    def train_step(self, images):
        with tf.GradientTape() as tape:
            loss, preds = self.compute_loss(images)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, preds

    def plot_images(self, batch, pred, epoch=None):
        num_cols = len(batch)
        num_rows = 2
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))

        for col in range(num_cols):
            plt.subplot(num_rows, num_cols, num_cols + col + 1)
            plt.imshow(batch[col])
            plt.subplot(num_rows, num_cols, col + 1)
            p_img = cv2.cvtColor((pred[col] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            plt.imshow(p_img)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"output_images/generated_plot_epoch-{epoch}.png")

    def convert_to_h5(self):
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder,
                                         vae=self.vae)
        manager = tf.train.CheckpointManager(checkpoint, directory='models/', max_to_keep=3)
        self.encoder.save('models/encoder_model.h5')
        self.decoder.save('models/decoder_model.h5')
        self.vae.save('models/vae_model.h5')

    def train(self):
        dataset = prepare_dataset(True)

        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.model.encoder,
                                         decoder=self.model.decoder,
                                         vae=self.model)
        manager = tf.train.CheckpointManager(checkpoint, directory='models/', max_to_keep=3)

        epoch_start = int(
            manager.latest_checkpoint.split(sep='ckpt-')[
                -1]) * save_every if manager.latest_checkpoint else 1
        print('starting at:', epoch_start)
        checkpoint.restore(manager.latest_checkpoint)
        bar = tf.keras.utils.Progbar(file_count * dataset_repetitions / batch_size - 1)
        for epoch, batch in enumerate(dataset):
            print(epoch)
        for epoch, batch in enumerate(dataset):
            loss, x_recon = self.train_step(batch)
            loss = np.array(loss)

            bar.update(epoch + epoch_start, values=[("loss", loss)])
            if (epoch + epoch_start) % save_every == 0:
                self.plot_images(batch, np.array(x_recon), epoch + epoch_start)

                manager.save()


g = VAE_interface()
g.train()
