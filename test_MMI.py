import os
import numpy as np
import tensorflow as tf
import shutil, sys

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from MMI import *
from utils import save_volume, data_IO, arg_parser
from utils.model import get_img_encoder, get_voxel_encoder, get_voxel_decoder
from utils import globals as g

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def main(args):

    weights_path = args.weights_file
    save_the_img = args.generate_img
    save_the_ori = args.save_ori
    voxel_data_path = args.voxel_data_dir
    image_data_path = args.image_data_dir
    input_form = args.input_form

    z_dim = args.latent_vector_size

    # Create new model that has only one input form and load specific part of trained weights
    if input_form == 'voxel':
        test_result_path = args.save_dir + '/test_sub_voxel_input'

        voxel_input = Input(shape=g.VOXEL_INPUT_SHAPE)
        voxel_encoder = get_voxel_encoder(z_dim)
        z = voxel_encoder(voxel_input)
        decoder = get_voxel_decoder(z_dim)
        output = decoder(voxel_encoder(voxel_input))
        model = Model(voxel_input, output, name='Voxel_VAE')
        model.load_weights(weights_path, by_name=True)

        voxel_test_data, hash = data_IO.voxelpath2matrix(voxel_data_path)
        reconstructions = model.predict(voxel_test_data)

    elif input_form == 'image':
        test_result_path = args.save_dir + '/test_sub_image_input'

        image_input = Input(shape=g.VIEWS_IMAGE_SHAPE)
        image_encoder = get_img_encoder(z_dim)

        z = image_encoder(image_input)
        decoder = get_voxel_decoder(z_dim)
        output = decoder(image_encoder(image_input))
        model = Model(image_input, output, name='Image_MVCNN_VAE')
        model.load_weights(weights_path, by_name=True)

        num_objects = len(os.listdir(image_data_path))
        images = np.zeros((num_objects,) + g.VIEWS_IMAGE_SHAPE, dtype=np.float32)
        object_files = os.listdir(image_data_path)
        hash = object_files

        for i, object in enumerate(object_files):
            image_path = os.path.join(image_data_path, object)
            images[i] = data_IO.imagepath2matrix(image_path)
        reconstructions = model.predict(images)


    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    # save the original test dataset file and generate the image
    if save_the_ori:
        voxel_path = voxel_data_path[:-9]
        ori_files_path = os.path.join(voxel_path,'test_sub_visulization')
        ori_files = os.listdir(ori_files_path)
        for file in ori_files:
            file = os.path.join(ori_files_path,file)
            shutil.copy2(file, test_result_path)

    # save the generated objects files
    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output_2(reconstructions[i, 0, :], hash[i], test_result_path, '_gen', save_bin= True, save_img= save_the_img)

if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))