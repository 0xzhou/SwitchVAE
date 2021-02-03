import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Conv3DTranspose, concatenate, Add, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from model import get_img_encoder,get_voxel_encoder, get_voxel_decoder
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

vol_input_shape = (1, 32, 32, 32)
img_input_shape= (12, 227, 227, 3)
img_shape = (227, 227, 3)
vol_weight = 0.5
img_weight = 0.5


def sample_latent_vectors(args):
    z_img, z_vol, w1, w2 = args
    return w1 * z_img + w2 * z_vol


def get_MMI(z_dim = 200):

    img_in = Input(shape= img_input_shape)
    vol_in = Input(shape= vol_input_shape)

    img_encoder = get_img_encoder(img_input_shape, z_dim, img_shape, img_in)
    vol_encoder = get_voxel_encoder(z_dim, vol_in)

    z_img = img_encoder(img_in)[2]
    z_vol = vol_encoder(vol_in)[2]

    weight_op_img = Lambda(lambda x: x * img_weight)
    weight_op_vol = Lambda(lambda x: x * vol_weight)

    weighted_z_img = weight_op_img(z_img)
    weighted_z_vol = weight_op_vol(z_vol)
    z = Add()([weighted_z_img, weighted_z_vol])


    encoder = Model([img_in, vol_in], z)
    plot_model(encoder, to_file='./MMI-encoder-test.pdf', show_shapes=True, expand_nested = True)

    dec_in = Input(shape=(200,))
    decoder, dec_output = get_voxel_decoder(dec_in)
    #out = decoder(z)
    decoder2 = Model(dec_in, dec_output)

    plot_model(decoder2, to_file='./MMI-decoder-test.pdf', show_shapes=True, expand_nested= True)

    output = decoder2(encoder([img_in,vol_in]))

    mmi = Model(inputs=[img_in, vol_in], outputs= output)
    #plot_model(mmi, to_file='./MMI.pdf', show_shapes=True, expand_nested=True)

    return { 'vol_inputs': vol_in,
            'img_inputs': img_in,
             'mmi': mmi
    }


if __name__ == '__main__':
    mmi = get_MMI(200)
    plot_model(mmi, to_file='./MMI.pdf', show_shapes=True, expand_nested= True)
