import numpy as np

import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Conv2D, MaxPool2D, Dense, Dropout, Flatten, Lambda, Reshape, Conv3DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from utils import globals as g

def sampling(args):
    mu, log_sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape = (batch, dim))

    return mu + K.exp(log_sigma) * epsilon

def get_voxel_encoder(z_dim=200):
    enc_in = Input(shape=g.VOXEL_INPUT_SHAPE, name='VoxEncoder_inputs')

    enc_conv1 = BatchNormalization(name='VoxEncoder_bn1')(Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                            padding='valid', kernel_initializer='glorot_normal', activation='elu',
                                            data_format='channels_first', name='VoxEncoder_conv1')(enc_in))

    enc_conv2 = BatchNormalization(name='VoxEncoder_bn2')(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                            padding='same', kernel_initializer='glorot_normal', activation='elu',
                                            data_format='channels_first', name='VoxEncoder_conv2')(enc_conv1))

    enc_conv3 = BatchNormalization(name='VoxEncoder_bn3')(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                            padding='valid', kernel_initializer='glorot_normal', activation='elu',
                                            data_format='channels_first', name='VoxEncoder_conv3')(enc_conv2))

    enc_conv4 = BatchNormalization(name='VoxEncoder_bn4')(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                            padding='same', kernel_initializer='glorot_normal', activation='elu',
                                            data_format='channels_first', name='VoxEncoder_conv4')(enc_conv3))

    enc_fc1 = BatchNormalization(name='VoxEncoder_bn_fc1')(Dense(units=343, kernel_initializer='glorot_normal',
                                         activation='elu', name='VoxEncoder_fcc1')(Flatten()(enc_conv4)))

    mu = BatchNormalization(name='VoxEncoder_bn_mu')(Dense(units=z_dim, kernel_initializer='glorot_normal',
                                    activation=None, name='VoxEncoder_mu')(enc_fc1))

    log_sigma = BatchNormalization(name='VoxEncoder_bn_log_sigma')(Dense(units=z_dim, kernel_initializer='glorot_normal',
                                           activation=None, name='VoxEncoder_log_sigma')(enc_fc1))

    z = Lambda(sampling, output_shape=(z_dim,), name='VoxEncoder_latent_vector')([mu, log_sigma])

    encoder = Model(enc_in, [mu, log_sigma, z], name='Voxel_Encoder')
    return encoder


def get_voxel_decoder(z_dim=200):
    dec_in = Input(shape=(z_dim,), name='VoxDecoder_inputs')

    dec_fc1 = BatchNormalization()(Dense(units=343, kernel_initializer='glorot_normal',
                                         activation='elu', name='VoxDecoder_fcc1', )(dec_in))

    dec_unflatten = Reshape(target_shape=(1, 7, 7, 7))(dec_fc1)

    dec_conv1 = BatchNormalization(name='VoxDecoder_bn1')(
        Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                        padding='same', kernel_initializer='glorot_normal',
                        activation='elu', name='VoxDecoder_conv1',
                        data_format='channels_first')(dec_unflatten))

    dec_conv2 = BatchNormalization(name='VoxDecoder_bn2')(
        Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                        padding='valid', kernel_initializer='glorot_normal',
                        activation='elu', name='VoxDecoder_conv2',
                        data_format='channels_first')(dec_conv1))

    dec_conv3 = BatchNormalization(name='VoxDecoder_bn3')(
        Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                        padding='same', kernel_initializer='glorot_normal',
                        activation='elu', name='VoxDecoder_conv3',
                        data_format='channels_first')(dec_conv2))

    dec_conv4 = BatchNormalization(name='VoxDecoder_bn4')(
        Conv3DTranspose(filters=8, kernel_size=(4, 4, 4), strides=(2, 2, 2),
                        padding='valid', kernel_initializer='glorot_normal',
                        activation='elu', name='VoxDecoder_conv4',
                        data_format='channels_first')(dec_conv3))

    dec_conv5 = BatchNormalization(beta_regularizer=l2(0.001), gamma_regularizer=l2(0.001), name='VoxDecoder_bn5') \
        (Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                         padding='same', kernel_initializer='glorot_normal',
                         data_format='channels_first', name='VoxDecoder_conv5', )(dec_conv4))

    decoder = Model(dec_in, dec_conv5, name='Voxel_Decoder')
    return decoder

def split_inputs(inputs):
    """
    split inputs to NUM_VIEW input
    :param inputs: a Input with shape VIEWS_IMAGE_SHAPE
    :return: a list of inputs which shape is IMAGE_SHAPE
    """
    slices = []
    for i in range(0, g.NUM_VIEWS):
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
    inputs = keras.Input(shape=input_shape, name='SVCNN_inputs')

    # this two layers don't omit any parameter for showing how to define conv and pool layer
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(3, 3),
                   padding='valid', activation='relu', use_bias=True,
                   kernel_initializer='glorot_normal', name='SVCNN_conv1')(inputs)
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid',
                      name='SVCNN_pool1')(conv1)

    conv2 = Conv2D(256, (5, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_normal', name='SVCNN_conv2')(pool1)
    pool2 = MaxPool2D((2, 2), (2, 2), name='SVCNN_pool2')(conv2)

    conv3 = Conv2D(384, (3, 3), padding='same', activation='relu',
                   kernel_initializer='glorot_normal', name='SVCNN_conv3')(pool2)
    conv4 = Conv2D(384, (3, 3), padding='same', activation='relu',
                   kernel_initializer='glorot_normal', name='SVCNN_conv4')(conv3)

    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu',
                   kernel_initializer='glorot_normal', name='SVCNN_conv5')(conv4)
    pool5 = MaxPool2D((2, 2), (2, 2), name='SVCNN_pool5')(conv5)

    reshape = Flatten(name='SVCNN_flatten1')(pool5)

    cnn = keras.Model(inputs=inputs, outputs=reshape, name='SVCNN')
    return cnn


def get_img_encoder(z_dim=200):
    """
    input: Batch x Viewns x Width x Height x Channels (tensor)
    """
    # input placeholder with shape (None, 12, 137, 137, 3)
    inputs = Input(shape=g.VIEWS_IMAGE_SHAPE, name='MVCNN_input')

    # split inputs into views(a list), every element of
    # view has shape (None, 137, 137, 3)
    views = Lambda(split_inputs, name='MVCNN_split')(inputs)
    cnn_model = cnn_img(g.IMAGE_SHAPE)
    view_pool = []

    # every view share the same cnn1_model(share the weights)
    for view in views:
        view_pool.append(cnn_model(view))

    # view pool layer
    pool5_vp = Lambda(_view_pool, name='MVCNN_view_pool')(view_pool)

    # cnn2 from here a full-connected layer
    fc6 = Dense(units=4096, activation='relu', kernel_regularizer=g.l2_reg, name='MVCNN_fcc6')(pool5_vp)

    # a dropout  layer, when call function evaluate and predict,
    # dropout layer will disabled automatically
    dropout6 = Dropout(0.6, name='MVCNN_dropout6')(fc6)

    fc7 = Dense(4096, 'relu', kernel_regularizer=g.l2_reg, name='MVCNN_fcc7')(dropout6)
    dropout7 = Dropout(0.6, name='MVCNN_dropout7')(fc7)
    fc8 = Dense(343, kernel_regularizer=g.l2_reg, name='MVCNN_fcc8')(dropout7)

    mu = BatchNormalization(name='MVCNN_bn_mu')(Dense(units=z_dim, kernel_initializer='glorot_normal',
                                    activation=None, name='MVCNN_mu')(fc8))
    log_sigma = BatchNormalization(name='MVCNN_bn_log_sigma')(Dense(units=z_dim, kernel_initializer='glorot_normal',
                                           activation=None, name='MVCNN_log_sigma')(fc8))

    z = Lambda(sampling, output_shape=(z_dim,), name='MVCNN_latent_vector')([mu, log_sigma])

    mvcnn_model = keras.Model(inputs=inputs, outputs=[mu, log_sigma, z], name='Image_MVCNN_VAE')
    return mvcnn_model

