import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import shutil, random, sys

from MMI import *
from utils import save_volume, data_IO, arg_parser, dataset_pre
from model import get_img_encoder, get_voxel_encoder, get_voxel_decoder

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def main(args):

    weights_path = args.weights_file
    test_result_path = args.save_dir + '/test1/'
    save_the_img = args.generate_img
    save_the_ori = args.save_ori
    voxel_data_path = args.voxel_data_dir
    image_data_path = args.image_data_dir
    input_form = args.input_form

    z_dim = args.latent_vector_size

    # Method1: Create new model that has only one input form and load specific part of trained weights
    if input_form == 'voxel':
        voxel_input = Input(shape=(1, 32, 32, 32))
        voxel_encoder = get_voxel_encoder(z_dim)
        z = voxel_encoder(voxel_input)
        decoder = get_voxel_decoder(z_dim)
        output = decoder(voxel_encoder(voxel_input))
        model = Model(voxel_input, output, name='Voxel_VAE')
        model.load_weights(weights_path, by_name=True)

        voxel_test_data, hash = data_IO.voxelpath2matrix(voxel_data_path)
        reconstructions = model.predict(voxel_test_data)

    elif input_form == 'image':
        image_input = Input(shape=(24, 137, 137, 3))
        image_encoder = get_img_encoder(input_shape=(24, 137, 137, 3), z_dim=z_dim, img_shape=(137, 137, 3),
                                        num_views=6)
        z = image_encoder(image_input)
        decoder = get_voxel_decoder(z_dim)
        output = decoder(image_encoder(image_input))
        model = Model(image_input, output, name='Image_MVCNN_VAE')
        model.load_weights(weights_path, by_name=True)

        num_objects = len(os.listdir(image_data_path))
        images = np.zeros((num_objects,) + (24, 137, 137, 3), dtype=np.float32)
        object_files = os.listdir(image_data_path)
        hash = object_files

        for i, object in enumerate(object_files):
            image_path = os.path.join(image_data_path, object)
            images[i] = data_IO.imagepath2matrix(image_path)
        reconstructions = model.predict(images)

    # Method2: Create MMI model and take some parts of it to build new model that has only one input form

    # MMI = get_MMI(z_dim, 'switch')['MMI']
    # MMI.load_weights(weights_path)
    #
    # # Create new model from MMI which takes only voxel or image as input
    # decoder = MMI.get_layer(name='Voxel_Generator')
    # if input_form == 'voxel':
    #     voxel_input = Input(shape=(1, 32, 32, 32))
    #     voxel_encoder = MMI.get_layer(name='Voxel_VAE')
    #     output = decoder(voxel_encoder(voxel_input))
    #     model = Model(voxel_input, output, name='test_voxel_VAE')
    #
    #     voxel_test_data, hash = data_IO.voxelpath2matrix(voxel_data_path)
    #     reconstructions = model.predict(voxel_test_data)
    #
    # elif input_form == 'image':
    #     image_input = Input(shape=(24, 137, 137, 3))
    #     image_encoder = MMI.get_layer(name='Image_MVCNN_VAE')
    #     output = decoder(image_encoder(image_input))
    #     model = Model(image_input, output, name='test_image_VAE')
    #
    #     num_objects = len(os.listdir(image_data_path))
    #     images = np.zeros((num_objects,) + (24, 137, 137, 3), dtype=np.float32)
    #     object_files = os.listdir(image_data_path)
    #     hash = object_files
    #
    #     for i, object in enumerate(object_files):
    #         image_path = os.path.join(image_data_path, object)
    #         images[i] = data_IO.imagepath2matrix(image_path)
    #     print("The shape of test image data:", images.shape)
    #     reconstructions = model.predict(images)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    # copy the original test dataset file
    if save_the_ori:
        for i in range(reconstructions.shape[0]):
            shutil.copy2('./dataset/03001627_test_sub/'+hash[i]+'.binvox', test_result_path)
            if save_the_img:
                shutil.copy2('./dataset/03001627_test_sub_images/' + hash[i] + '.png', test_result_path)

    # save the generated objects files
    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output(reconstructions[i, 0, :], hash[i], test_result_path, '_gen', save_bin= True, save_img= save_the_img)

if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))