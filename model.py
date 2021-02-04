import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Conv2D, MaxPool2D, Dense, Dropout, Flatten, Lambda, Reshape, Conv3DTranspose
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
channels = 3
xavier = keras.initializers.glorot_normal()
l2_reg = keras.regularizers.l2(0.004)

#image_input_shape = (views, image_width, image_length, channels)

def sampling(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape = (batch, dim))

    return mu + K.exp(0.5 * sigma) * epsilon

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

    encoder = Model(enc_in, [mu, sigma, z], name='Voxel_VAE')
    #encoder = Model(vol_input, [mu, sigma, z])
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

    decoder = Model(dec_in, dec_conv5, name= 'Voxel_Generator')
    return decoder

def split_inputs(inputs, num_views=6):
    """
    split inputs to NUM_VIEW input
    :param inputs: a Input with shape VIEWS_IMAGE_SHAPE
    :return: a list of inputs which shape is IMAGE_SHAPE
    """
    slices = []
    for i in range(0, num_views):
        slices.append(inputs[:, i, :, :, :])
    return slices

def _view_pool(views):
    """
    this is the ViewPooling in the paper
    :param views: the NUM_VIEWS outputs of CNN1
    """
    expanded = [K.expand_dims(view, 0) for view in views]
    concated = K.concatenate(expanded, 0)
    reduced = K.max(concated, 0)
    return reduced

def cnn_img(input_shape):
    """
       this is the CNN1 Network in paper
       :param input_shape: a image's shape, not the shape of a batch of image
       :return: a model object
       """
    inputs = keras.Input(shape=input_shape, name='inputs')

    # this two layers don't omit any parameter for showing how to define conv and pool layer
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(3, 3),
                   padding='valid', activation='relu', use_bias=True,
                   kernel_initializer=xavier, name='conv1')(inputs)
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid',
                      name='pool1')(conv1)

    # we omit some default parameters
    conv2 = Conv2D(256, (5, 5), padding='same', activation='relu',
                   kernel_initializer=xavier, name='conv2')(pool1)
    pool2 = MaxPool2D((2, 2), (2, 2), name='pool2')(conv2)

    conv3 = Conv2D(384, (3, 3), padding='same', activation='relu',
                   kernel_initializer=xavier, name='conv3')(pool2)
    conv4 = Conv2D(384, (3, 3), padding='same', activation='relu',
                   kernel_initializer=xavier, name='conv4')(conv3)
    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu',
                   kernel_initializer=xavier, name='conv5')(conv4)

    pool5 = MaxPool2D((2, 2), (2, 2), name='pool5')(conv5)

    reshape = Flatten(name='reshape')(pool5)

    cnn = keras.Model(inputs=inputs, outputs=reshape, name='cnn')
    return cnn


def get_img_encoder(input_shape=None, z_dim = 200, img_shape=None):
    """
    input: Batch x Viewns x Width x Height x Channels (tensor)
    """
    # input placeholder with shape (None, 12, 227, 227, 3)
    # 'None'=batch size; 12=NUM_VIEWS; (227, 227, 3)=IMAGE_SHAPE

    inputs = Input(shape=input_shape, name='input')

    # split inputs into views(a list), every element of
    # view has shape (None, 227, 227, 3).
    views = Lambda(split_inputs, name='split')(inputs)
    #views = Lambda(split_inputs, name='split')(img_inputs)

    cnn_model = cnn_img(img_shape)

    view_pool = []
    # every view share the same cnn1_model(share the weights)
    for view in views:
        view_pool.append(cnn_model(view))

    # view pool layer
    pool5_vp = Lambda(_view_pool, name='view_pool')(view_pool)

    # cnn2 from here
    # a full-connected layer
    fc6 = Dense(units=4096, activation='relu',
                kernel_regularizer=l2_reg, name='fc6')(pool5_vp)
    # a dropout layer, when call function evaluate and predict,
    # dropout layer will disabled automatically
    dropout6 = Dropout(0.6, name='dropout6')(fc6)

    fc7 = Dense(4096, 'relu', kernel_regularizer=l2_reg, name='fc7')(dropout6)
    dropout7 = Dropout(0.6, name='dropout7')(fc7)

    fc8 = Dense(343, kernel_regularizer=l2_reg, name='fc8')(dropout7)

    mu = BatchNormalization()(
        Dense(
            units = z_dim,
            kernel_initializer = 'glorot_normal',
            activation = None)(fc8))
    sigma = BatchNormalization()(
        Dense(
            units = z_dim,
            kernel_initializer = 'glorot_normal',
            activation = None)(fc8))
    z = Lambda(
        sampling,
        output_shape = (z_dim, ))([mu, sigma])

    mvcnn_model = keras.Model(inputs=inputs, outputs=[mu, sigma, z], name='Image_MVCNN_VAE')
    return mvcnn_model



# class VAE(keras.Model):
#     def __init__(self, beta=1, tc=False, latent_dims=200, **kwargs):
#         super(VAE, self).__init__(**kwargs)
#         self.latent_dims = latent_dims
#         self.encoder = get_voxel_encoder(self.latent_dims)
#         self.decoder = get_voxel_decoder(self.latent_dims)
#         self.beta = beta
#         self.tc = tc
#
#
#     def call(self, input):
#         return None
#
#     def train_step(self, data):
#         if isinstance(data, tuple):
#             data = data[0]
#
#         with tf.GradientTape() as tape:
#             mu, log_sigma_square, z = self.encoder(data)[0], self.encoder(data)[1], self.encoder(data)[2]
#             px_z = self.decoder(z)
#
#             BCE_loss = K.cast(K.mean(custom_loss.weighted_binary_crossentropy(data, K.clip(sigmoid(px_z), 1e-7, 1.0 - 1e-7))), 'float32')
#             kl_loss_term = custom_loss.kl_loss(mu, log_sigma_square)
#             tc = total_correlation(z, mu, log_sigma_square)
#             tc_loss_term = (self.beta - 1.) * tc
#             #tc_loss_term, tc = custom_loss.tc_term(self.beta, z, mu, log_sigma_square)
#
#             if self.tc:
#                 elbo = BCE_loss + kl_loss_term + tc_loss_term
#             else:
#                 elbo = BCE_loss + self.beta * kl_loss_term
#
#             total_loss = elbo
#             grads = tape.gradient(total_loss, self.trainable_weights)
#             self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#
#             return {
#                 'loss': total_loss,
#                 'bce_loss': BCE_loss,
#                 'kl': kl_loss_term,
#                 'tc': tc
#             }
#
#     def save(self, save_path):
#         self.encoder.save(os.path.join(save_path, 'encoder'))
#         self.decoder.save(os.path.join(save_path, 'decoder'))


# def total_correlation(zs, mu, sigma):
#
#     log_qz_prob = tfp.distributions.Normal(tf.expand_dims(mu, 0), tf.expand_dims(sigma, 0)).log_prob(tf.expand_dims(zs, 1))
#     log_qz_prod = tf.reduce_sum(tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False), axis=1, keepdims=False)
#     log_qz = tf.reduce_logsumexp(tf.reduce_sum(log_qz_prob, axis=2, keepdims=False), axis=1, keepdims=False)
#
#     return log_qz - log_qz_prod