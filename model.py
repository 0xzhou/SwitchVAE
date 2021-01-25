import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Conv3DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

from tensorflow.keras.activations import sigmoid
from utils import custom_loss
import os

voxel_input_shape = (1, 32, 32, 32)
views = 6
image_width = 128
image_length = 128
channels =3

image_input_shape = (views, image_width, image_length, channels)

class VAE(keras.Model):
    def __init__(self, beta=1, tc=False, latent_dims=200, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dims = latent_dims
        self.encoder = get_voxel_encoder(self.latent_dims)
        self.decoder = get_voxel_decoder(self.latent_dims)
        self.beta = beta
        self.tc = tc

    def call(self, input):
        return None

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            mu, log_sigma_square, z = self.encoder(data)[0], self.encoder(data)[1], self.encoder(data)[2]
            px_z = self.decoder(z)

            BCE_loss = K.cast(K.mean(custom_loss.weighted_binary_crossentropy(data, K.clip(sigmoid(px_z), 1e-7, 1.0 - 1e-7))), 'float32')
            kl_loss = custom_loss.kl_loss(mu, log_sigma_square)
            tc = total_correlation(z, mu, log_sigma_square)

            if self.tc:
                elbo = BCE_loss + kl_loss + (self.beta -1.) * tc
            else:
                elbo = BCE_loss + self.beta * kl_loss

            total_loss = elbo
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                'loss': total_loss,
                'bce_loss': BCE_loss,
                'kl': kl_loss,
                'tc': tc
            }

    def save(self, save_path):
        self.encoder.save(os.path.join(save_path, 'encoder'))
        self.decoder.save(os.path.join(save_path, 'decoder'))

def sampling(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape = (batch, dim))

    return mu + K.exp(0.5 * sigma) * epsilon

def total_correlation(zs, mu, sigma):

    log_qz_prob = tfp.distributions.Normal(tf.expand_dims(mu, 0), tf.expand_dims(sigma, 0)).log_prob(tf.expand_dims(zs, 1))
    log_qz_prod = tf.reduce_sum(tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False), axis=1, keepdims=False)
    log_qz = tf.reduce_logsumexp(tf.reduce_sum(log_qz_prob, axis=2, keepdims=False), axis=1, keepdims=False)

    return log_qz - log_qz_prod

def get_voxel_encoder(z_dim = 200):
    enc_in = Input(shape=voxel_input_shape)

    enc_conv1 = BatchNormalization()(
        Conv3D(
            filters=8,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='valid',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(enc_in))
    enc_conv2 = BatchNormalization()(
        Conv3D(
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(enc_conv1))
    enc_conv3 = BatchNormalization()(
        Conv3D(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='valid',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(enc_conv2))
    enc_conv4 = BatchNormalization()(
        Conv3D(
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(enc_conv3))

    enc_fc1 = BatchNormalization()(
        Dense(
            units=343,
            kernel_initializer='glorot_normal',
            activation='elu')(Flatten()(enc_conv4)))
    mu = BatchNormalization()(
        Dense(
            units=z_dim,
            kernel_initializer='glorot_normal',
            activation=None)(enc_fc1))
    sigma = BatchNormalization()(
        Dense(
            units=z_dim,
            kernel_initializer='glorot_normal',
            activation=None)(enc_fc1))
    z = Lambda(
        sampling,
        output_shape=(z_dim,))([mu, sigma])
    encoder = Model(enc_in, [mu, sigma, z])
    return encoder

def get_voxel_decoder(z_dim = 200):
    dec_in = Input(shape=(z_dim,))

    dec_fc1 = BatchNormalization()(
        Dense(
            units=343,
            kernel_initializer='glorot_normal',
            activation='elu')(dec_in))
    dec_unflatten = Reshape(
        target_shape=(1, 7, 7, 7))(dec_fc1)

    dec_conv1 = BatchNormalization()(
        Conv3DTranspose(
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(dec_unflatten))
    dec_conv2 = BatchNormalization()(
        Conv3DTranspose(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='valid',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(dec_conv1))
    dec_conv3 = BatchNormalization()(
        Conv3DTranspose(
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(dec_conv2))
    dec_conv4 = BatchNormalization()(
        Conv3DTranspose(
            filters=8,
            kernel_size=(4, 4, 4),
            strides=(2, 2, 2),
            padding='valid',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(dec_conv3))
    dec_conv5 = BatchNormalization(
        beta_regularizer=l2(0.001),
        gamma_regularizer=l2(0.001))(
        Conv3DTranspose(
            filters=1,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            kernel_initializer='glorot_normal',
            data_format='channels_first')(dec_conv4))

    decoder = Model(dec_in, dec_conv5)
    return decoder

def get_img_encoder(inputs, keep_prob=0.5, phase_train=True, reuse=False):
    """
    input: Batch x Viewns x Width x Height x Channels (tensor)
    """
    # transpose views : (BxVxWxHxC) -> (VxBxWxHxC)
    n_views = inputs.get_shape().as_list()[1]
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    view_pool = []

    for i in range(n_views):
        # set reuse True for i > 0, for weight-sharing
        reuse = reuse or (i != 0)
    with tf.variable_scope("I-encoder", reuse=reuse):
        view = tf.gather(views, i)  # BxWxHxC B*128*128*3

    conv1 = tf.nn.conv2d(view, weights['weI1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=phase_train)
    conv1 = tf.nn.relu(conv1)  # B*128*128*64

    conv2_1 = tf.nn.conv2d(conv1, weights['weB2_1'], strides=[1, 2, 2, 1], padding='SAME')
    conv2_2 = tf.nn.conv2d(conv2_1, weights['weB2_2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_3 = tf.nn.conv2d(conv2_2, weights['weB2_3'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.concat([conv2_1, conv2_3], axis=-1)
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=phase_train)
    conv2 = tf.nn.relu(conv2)  # B*64*64*128

    conv3_1 = tf.nn.conv2d(conv2, weights['weB3_1'], strides=[1, 2, 2, 1], padding='SAME')
    conv3_2 = tf.nn.conv2d(conv3_1, weights['weB3_2'], strides=[1, 1, 1, 1], padding='SAME')
    conv3_3 = tf.nn.conv2d(conv3_2, weights['weB3_3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.concat([conv3_1, conv3_3], axis=-1)
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=phase_train)
    conv3 = tf.nn.relu(conv3)  # B*32*32*256

    conv4_1 = tf.nn.conv2d(conv3, weights['weB4_1'], strides=[1, 2, 2, 1], padding='SAME')
    conv4_2 = tf.nn.conv2d(conv4_1, weights['weB4_2'], strides=[1, 1, 1, 1], padding='SAME')
    conv4_3 = tf.nn.conv2d(conv4_2, weights['weB4_3'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.concat([conv4_1, conv4_3], axis=-1)
    conv4 = tf.contrib.layers.batch_norm(conv4, is_training=phase_train)
    conv4 = tf.nn.relu(conv4)  # B*16*16*512

    conv5_1 = tf.nn.conv2d(conv4, weights['weB5_1'], strides=[1, 2, 2, 1], padding='SAME')
    conv5_2 = tf.nn.conv2d(conv5_1, weights['weB5_2'], strides=[1, 1, 1, 1], padding='SAME')
    conv5_3 = tf.nn.conv2d(conv5_2, weights['weB5_3'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.concat([conv5_1, conv5_3], axis=-1)
    conv5 = tf.contrib.layers.batch_norm(conv5, is_training=phase_train)
    conv5 = tf.nn.relu(conv5)  # B*8*8*1024

    pool6 = tf.nn.avg_pool(conv5, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')  # B*1*1*1024
    # dim = np.prod(pool5.get_shape().as_list()[1:])
    # reshape = tf.reshape(pool5, [-1, dim])
    view_pool.append(pool6)

    # view_pool = tf.reshape(view_pool, (batch_size, 1, 1, -1, n_views)) #B*1*1*1024*V
    view_pool = tf.stack(view_pool, 0)
    view_pool = tf.transpose(view_pool, perm=[1, 2, 3, 4, 0])  # B*1*1*1024*V
    pool6_vp = tf.reduce_max(view_pool, axis=-1)  # B*1*1*1024

with tf.variable_scope("encoderB", reuse=reuse):
    fc7 = tf.nn.conv2d(pool6_vp, weights['weB7'], strides=[1, 1, 1, 1], padding='SAME')  # B*1*1*512
    fc8 = tf.nn.conv2d(fc7, weights['weB8'], strides=[1, 1, 1, 1], padding='SAME')  # B*1*1*z_size
    fc8 = tf.nn.tanh(fc8)

return fc8


