from tensorflow import exp, reduce_sum,pow
from config import variational_beta
import tensorflow as tf



def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    BCE = tf.keras.losses.BinaryCrossentropy(reduction='sum')
    recon_loss = BCE(recon_x, x)

    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * reduce_sum(1 + logvar - pow(mu, 2) - exp(logvar))

    return recon_loss + variational_beta * kldivergence
