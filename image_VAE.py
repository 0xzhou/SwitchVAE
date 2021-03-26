import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from utils.model import get_img_encoder, get_voxel_decoder
import utils.globals as g


def get_image_VAE(z_dim=200, use_pretrain=True):
    #img_input = Input(shape=g.VIEWS_IMAGE_SHAPE_MODELNET, name='Image_Input')
    img_input = Input(shape=g.VIEWS_IMAGE_SHAPE_SHAPENET, name='Image_Input')
    vol_input = Input(shape=g.VOXEL_INPUT_SHAPE, name='Voxel_Input')

    img_encoder_model = get_img_encoder(z_dim)
    img_encoder = img_encoder_model['image_encoder']
    image_embedding_model = img_encoder_model['image_embedding_model']
    view_feature_aggregator = img_encoder_model['view_feature_aggregator']

    if use_pretrain:
        image_embedding_model.load_weights('./utils/resnet18_imagenet_1000_no_top.h5', by_name=True)

    img_z_mean, img_z_logvar, img_z = img_encoder(img_input)
    decoder = get_voxel_decoder(z_dim)
    recontructions = decoder(img_z)
    original_voxel = vol_input

    img_vae = Model([img_input, vol_input], [recontructions, original_voxel], name='Image_VAE')

    return {'image_inputs': img_input,
            'vol_inputs': vol_input,
            'outputs': recontructions,
            'img_z_mean': img_z_mean,
            'img_z_logvar': img_z_logvar,
            'img_z': img_z,
            'image_encoder': img_encoder,
            'image_embedding_model': image_embedding_model,
            'view_feature_aggregator':view_feature_aggregator,
            'image_vae_decoder': decoder,
            'image_vae': img_vae
            }
