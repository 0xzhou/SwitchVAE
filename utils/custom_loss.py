import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


def weighted_binary_crossentropy(target, output):
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss


def kl_loss(z_mean, z_log_sigma):
    return - 0.5 * K.mean(1 + 2 * z_log_sigma - K.square(z_mean) - K.exp(2 * z_log_sigma))


def tc_term(beta, z_sampled, z_mean, z_log_squared_scale):
    """
    From:
    Locatello, F. et al.
    Challenging Common Assumptions in the Unsupervised Learning
    of Disentangled Representations. (2018).
    Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
    Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.
    --
    :param args: Shared arguments
    :param z_sampled: Samples from latent space
    :param z_mean: Means of z
    :param z_log_squared_scale: Logvars of z
    :return: Total correlation penalty
    """
    tc = total_correlation(z_sampled, z_mean, z_log_squared_scale, 'normal')
    return (beta - 1.) * tc, tc


def gaussian_log_density(samples, mean, log_squared_scale):
    pi = tf.constant(np.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-log_squared_scale)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_squared_scale + normalization)


def total_correlation(z, z_mean, z_log_squared_scale, prior):
    """Estimate of total correlation on a batch.
    We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
    for the minimization. The constant should be equal to (num_latents - 1) *
    log(batch_size * dataset_size)
    Args:
      z: [batch_size, num_latents]-tensor with sampled representation.
      z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
      z_log_squared_scale: [batch_size, num_latents]-tensor with log variance of the encoder.
    Returns:
      Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    if prior.lower() == "normal":
        log_qz_prob = gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_log_squared_scale, 0))
    # if prior.lower() == "laplace":
    #     log_qz_prob = laplace_log_density(
    #         tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
    #         tf.expand_dims(z_log_squared_scale, 0))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = tf.math.reduce_sum(
        tf.math.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
        axis=1,
        keepdims=False)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = tf.math.reduce_logsumexp(
        tf.math.reduce_sum(log_qz_prob, axis=2, keepdims=False),
        axis=1,
        keepdims=False)
    return K.mean(log_qz - log_qz_product)
