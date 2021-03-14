import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


def weighted_binary_crossentropy(target, output):
    loss = -(80.0 * target * K.log(output) + 20.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    loss = K.mean(K.sum(loss, axis=(-3, -2, -1)))
    return loss


def kl_loss(z_mean, z_logvar):
    loss = -0.5 * (1 + z_logvar - K.square(z_mean) - K.exp(z_logvar))
    loss = K.mean(K.sum(loss, axis=1))
    return loss


def gaussian_log_density(samples, mean, logvar):

    pi = tf.constant(np.pi)
    normalization = tf.log(2. * pi)
    inv_sigma = tf.exp(-logvar)
    tmp = (samples - mean)
    result = -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)
    return result


def MSE(v1, v2):
    loss = K.square(v1 - v2)
    loss = K.mean(K.sum(loss, axis=1))
    return loss


def total_correlation(z, z_mean, z_logvar, prior='normal'):
    """Estimate of total correlation on a batch.
    We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
    for the minimization. The constant should be equal to (num_latents - 1) *
    log(batch_size * dataset_size)
    Args:
      z: [batch_size, num_latents]-tensor with sampled representation.
      z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
      z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
    Returns:
      Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    if prior.lower() == "normal":
        log_qz_prob = gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_logvar, 0))
    # if prior.lower() == "laplace":
    #     log_qz_prob = laplace_log_density(
    #         tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
    #         tf.expand_dims(z_log_squared_scale, 0))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = tf.reduce_sum(
        tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
        axis=1,
        keepdims=False)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = tf.reduce_logsumexp(
        tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
        axis=1,
        keepdims=False)
    return tf.reduce_mean(log_qz - log_qz_product)
