from config import dataset_repetitions, batch_size
from model import log_normal_pdf, VAE
from keras.optimizers import Adam
from data import prepare_dataset
import matplotlib.pyplot as plt
from config import save_every
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import Tensor
from numpy import ndarray
from typing import Tuple
import tensorflow as tf
import numpy as np

file_count = sum(len(files) for _, _, files in os.walk(r'data/images'))


class VAEInterface:
    def __init__(self):
        self.model = VAE()
        self.optimizer = Adam(learning_rate=1e-4)

    def compute_loss(self, x: ndarray) -> Tuple[Tensor, Tensor]:
        mean, logvar = self.model.encode(x)
        z = self.model.reparameterize(mean, logvar)
        x_logit = self.model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        log_px_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        log_pz = log_normal_pdf(z, 0., 0.)
        log_qz_x = log_normal_pdf(z, mean, logvar)
        loss = -tf.reduce_mean(log_px_z + log_pz - log_qz_x)
        return loss, x_logit

    def train_step(self, images: ndarray) -> Tuple[Tensor, Tensor]:
        with tf.GradientTape() as tape:
            loss, predicts = self.compute_loss(images)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, predicts

    def plot_images(self, batch: ndarray, predicts: Tensor, epoch: bool = None) -> None:
        predicts = np.array(tf.sigmoid(predicts))
        num_cols = len(batch)
        num_rows = 2
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))

        for col in range(num_cols):
            plt.subplot(num_rows, num_cols, num_cols + col + 1)
            plt.imshow(batch[col])
            plt.subplot(num_rows, num_cols, col + 1)

            plt.imshow((predicts[col] * 255).astype(np.uint8))
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"output_images/generated_plot_epoch-{epoch}.png")

    def generate_image(self, images: ndarray) -> None:
        mean, log_var = self.model.encode(images)
        z = self.model.reparameterize(mean, log_var)
        predicts = self.model.decode(z)
        predicts = np.array(tf.sigmoid(predicts))
        plt.imshow((predicts * 255).astype(np.uint8))

    def convert_to_h5(self) -> None:
        self.model.encoder.save('models/encoder_model.h5')
        self.model.decoder.save('models/decoder_model.h5')

    def train(self):
        dataset = prepare_dataset(True)

        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.model.encoder,
                                         decoder=self.model.decoder,
                                         vae=self.model)
        manager = tf.train.CheckpointManager(checkpoint, directory='models/', max_to_keep=3)

        epoch_start = int(
            manager.latest_checkpoint.split(sep='ckpt-')[
                -1]) * save_every + 1 if manager.latest_checkpoint else 1
        print('starting at:', epoch_start)
        checkpoint.restore(manager.latest_checkpoint)
        bar = tf.keras.utils.Progbar(file_count * dataset_repetitions / batch_size - 1)

        for epoch, batch in enumerate(dataset):
            loss, x_recon = self.train_step(batch)
            loss = np.array(loss)

            bar.update(epoch + epoch_start, values=[("loss", loss)])
            if (epoch + epoch_start) % save_every == 0:
                self.plot_images(batch, x_recon, epoch + epoch_start)

                manager.save()
