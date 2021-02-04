import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Conv2D, Dense, Flatten, Lambda, Reshape, MaxPool2D, Dropout, Conv3DTranspose, concatenate, Add, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from model import get_img_encoder, get_voxel_encoder, get_voxel_decoder
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import random

voxel_input_shape = (1, 32, 32, 32)
img_input_shape= (24, 137, 137, 3)
img_shape = (137, 137, 3)
vol_weight = 0.5
img_weight = 0.5
xavier = keras.initializers.glorot_normal()
l2_reg = keras.regularizers.l2(0.004)


def switch(args):
    z_img, z_vol = args
    switch_possibility = random.random()
    if switch_possibility > 0.5:
        return z_img + 0.00001 * z_vol
    else:
        return z_vol + 0.00001 * z_img

def get_MMI(z_dim = 200, train_mode = None):

    img_input = Input(shape= img_input_shape, name='Image_Input')
    vol_input = Input(shape= voxel_input_shape, name='Voxel_Input')

    img_encoder = get_img_encoder(img_input_shape, z_dim, img_shape)
    vol_encoder = get_voxel_encoder(z_dim)

    mu_img, mu_vol = img_encoder(img_input)[0], vol_encoder(vol_input)[0]
    sigma_img, sigma_vol = img_encoder(img_input)[1], vol_encoder(vol_input)[1]
    z_img, z_vol= img_encoder(img_input)[2], vol_encoder(vol_input)[2]

    # Method1: Use "Switch" to train the latent vectors
    if train_mode == 'switch':
        z = Lambda(switch, output_shape=(z_dim,), name= 'Switch_Layer')([z_img, z_vol])

    # Method2: Add latent vectors from different input with weights to generate the latent vectors
    elif train_mode == 'weighted_add':
        weight_op_img = Lambda(lambda x: x * img_weight, name='Imgae_Weighted_Layer')
        weight_op_vol = Lambda(lambda x: x * vol_weight, name='Voxel_Weighted_Layer')

        weighted_z_img = weight_op_img(z_img)
        weighted_z_vol = weight_op_vol(z_vol)

        z = Add(name='Weighted_Add_Layer')([weighted_z_img, weighted_z_vol])

    # Method3: Use a full connect layer to generated the latent vectors
    elif train_mode == 'fcc':
        z = concatenate([z_img, z_vol])
        z = Dense(units=z_dim, activation= 'tanh', name='FCC_Layer')(z)


    MMI_encoder = Model([img_input, vol_input], z)
    #plot_model(MMI_encoder, to_file='./MMI_encoder.pdf', show_shapes=True, expand_nested = True)

    MMI_decoder = get_voxel_decoder(z_dim)
    decoded_vol = MMI_decoder(z)

    #plot_model(MMI_encoder, to_file='./MMI-decoder.pdf', show_shapes=True, expand_nested=True)

    MMI = Model(MMI_encoder.inputs, decoded_vol)
    #plot_model(MMI, to_file='./MMI.pdf', show_shapes=True, expand_nested=True)
    #plot_model(MMI, to_file='./nested-MMI.pdf', show_shapes=True)

    return { 'vol_inputs': vol_input,
             'img_inputs': img_input,
             'mu_img': mu_img,
             'mu_vol': mu_vol,
             'sigma_img': sigma_img,
             'sigma_vol': sigma_vol,
             'z_img': z_img,
             'z_vol': z_vol,
             'z': z,
             'MMI_encoder': MMI_encoder,
             'MMI_decoder': MMI_decoder,
             'MMI': MMI,
             'outputs': decoded_vol
    }

if __name__ == '__main__':
    get_MMI(200, 'switch')


