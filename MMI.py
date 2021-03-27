import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from utils.model import get_img_encoder, get_voxel_encoder, get_voxel_decoder
import utils.globals as g


def switch(args):
    img_output, vol_output = args[0], args[1]
    switch = K.random_uniform(shape=(1,))
    switch = tf.floor(switch / g.SWITCH_PROBABILITY)
    output = switch * img_output + (1 - switch) * vol_output
    return output[0], output[1], output[2]


def get_MMI(z_dim=200, view_image_shape=None, train_mode='switch', use_pretrain=True):

    img_input = Input(shape=view_image_shape, name='Image_Input')
    vol_input = Input(shape=g.VOXEL_INPUT_SHAPE, name='Voxel_Input')

    img_encoder_model = get_img_encoder(z_dim, view_image_shape)
    img_encoder = img_encoder_model['image_encoder']
    image_embedding_model = img_encoder_model['image_embedding_model']
    view_feature_aggregator = img_encoder_model['view_feature_aggregator']
    vol_encoder = get_voxel_encoder(z_dim)

    if use_pretrain:
        image_embedding_model.load_weights('./utils/resnet18_imagenet_1000_no_top.h5', by_name=True)

    img_encoder_output = img_encoder(img_input)
    vol_encoder_output = vol_encoder(vol_input)

    # Method1: Use "Switch" to train the latent vectors
    if train_mode == 'switch':
        z_mean, z_logvar, z = Lambda(switch, output_shape=(z_dim,), name='Switch_Layer')([img_encoder_output, vol_encoder_output])

    # Method2: Add latent vectors from different input with weights to generate the latent vectors
    elif train_mode == 'weighted_add':
        weight_op_img = Lambda(lambda x: x * g.IMG_WEIGHT, name='Imgae_Weighted_Layer')
        weight_op_vol = Lambda(lambda x: x * g.VOL_WEIGHT, name='Voxel_Weighted_Layer')

        img_z_mean, img_z_logvar, img_z = [weight_op_img(x) for x in img_encoder_output]
        vol_z_mean, vol_z_logvar, vol_z = [weight_op_vol(x) for x in vol_encoder_output]
        z_mean = Add(name='Weighted_Add_z_mean')([img_z_mean, vol_z_mean])
        z_logvar = Add(name='Weighted_Add_z_logvar')([img_z_logvar, vol_z_logvar])
        z = Add(name='Weighted_Add_z')([img_z, vol_z])

    # Method3: Use a full connect layer to generated the latent vectors
    # elif train_mode == 'fcc':
    #     z = concatenate([z_img, z_vol])
    #     z = Dense(units=z_dim, activation= 'tanh', name='FCC_Layer')(z)

    MMI_encoder = Model([img_input, vol_input], [z_mean, z_logvar, z])
    MMI_decoder = get_voxel_decoder(z_dim)
    decoded_vol = MMI_decoder(z)

    MMI = Model([img_input, vol_input], decoded_vol)

    return {'vol_inputs': vol_input,
            'img_inputs': img_input,
            'img_z_mean': img_encoder_output[0],
            'vol_z_mean': vol_encoder_output[0],
            'img_z_logvar': img_encoder_output[1],
            'vol_z_logvar': vol_encoder_output[1],
            'z_img': img_encoder_output[2],
            'z_vol': vol_encoder_output[2],
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'z': z,
            'image_encoder': img_encoder,
            'image_embedding_model': image_embedding_model,
            'view_feature_aggregator': view_feature_aggregator,
            'voxel_encoder': vol_encoder,
            'MMI_encoder': MMI_encoder,
            'MMI_decoder': MMI_decoder,
            'MMI': MMI,
            'outputs': decoded_vol,
            }
