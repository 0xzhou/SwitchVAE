import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Conv3DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import utils.globals as g


def sampling(args):
    z_mean, z_logvar = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(z_logvar / 2.0) * epsilon

def get_voxel_VAE(z_dim = 200):
    enc_in = Input(shape=g.VOXEL_INPUT_SHAPE, name='VoxEncoder_inputs')

    enc_conv1 = BatchNormalization(name='VoxEncoder_bn1')(Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                                                 padding='valid', kernel_initializer='glorot_normal',
                                                                 activation='elu',
                                                                 data_format='channels_first', name='VoxEncoder_conv1')(
        enc_in))

    enc_conv2 = BatchNormalization(name='VoxEncoder_bn2')(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                                                 padding='same', kernel_initializer='glorot_normal',
                                                                 activation='elu',
                                                                 data_format='channels_first', name='VoxEncoder_conv2')(
        enc_conv1))

    enc_conv3 = BatchNormalization(name='VoxEncoder_bn3')(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                                                 padding='valid', kernel_initializer='glorot_normal',
                                                                 activation='elu',
                                                                 data_format='channels_first', name='VoxEncoder_conv3')(
        enc_conv2))

    enc_conv4 = BatchNormalization(name='VoxEncoder_bn4')(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                                                 padding='same', kernel_initializer='glorot_normal',
                                                                 activation='elu',
                                                                 data_format='channels_first', name='VoxEncoder_conv4')(
        enc_conv3))

    enc_fc1 = BatchNormalization(name='VoxEncoder_bn_fc1')(Dense(units=343, kernel_initializer='glorot_normal',
                                                                 activation='elu', name='VoxEncoder_fc1')(
        Flatten(name='VoxEncoder_flatten1')(enc_conv4)))

    z_mean = BatchNormalization(name='VoxEncoder_bn_z_mean')(Dense(units=z_dim, kernel_initializer='glorot_normal',
                                                                   activation=None, name='VoxEncoder_z_mean')(enc_fc1))

    z_logvar = BatchNormalization(name='VoxEncoder_bn_z_logvar')(Dense(units=z_dim, kernel_initializer='glorot_normal',
                                                                       activation=None, name='VoxEncoder_z_logvar')(
        enc_fc1))

    z = Lambda(sampling, output_shape=(z_dim,), name='VoxEncoder_z')([z_mean, z_logvar])

    encoder = Model(enc_in, [z_mean, z_logvar, z], name='Voxel_Encoder')

    dec_in = Input(shape=(z_dim,), name='VoxDecoder_inputs')

    dec_fc1 = BatchNormalization(name='VoxDecoder_bn_fc1')(Dense(units=343, kernel_initializer='glorot_normal',
                                                                 activation='elu', name='VoxDecoder_fc1')(dec_in))

    dec_unflatten = Reshape(target_shape=(1, 7, 7, 7), name='VoxDecoder_reshape1')(dec_fc1)

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

    dec_conv5 = decoder(encoder(enc_in)[2])

    vae = Model(enc_in, dec_conv5,name='Voxel_VAE')

    return {'inputs': enc_in, 
            'outputs': dec_conv5,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'z': z,
            'encoder': encoder,
            'decoder': decoder,
            'vae': vae}
