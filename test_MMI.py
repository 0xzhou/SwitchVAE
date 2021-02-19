import os
import numpy as np
import tensorflow as tf
import shutil, sys

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from utils import save_volume, data_IO, arg_parser, model
from utils import globals as g
from MMI import *

ConFig = tf.ConfigProto()
ConFig.gpu_options.allow_growth = True
session = tf.Session(config=ConFig)


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
        voxel_encoder = model.get_voxel_encoder(z_dim)
        decoder = model.get_voxel_decoder(z_dim)
        output = decoder(voxel_encoder(voxel_input))
        voxel_vae = Model(voxel_input, output, name='Test_Voxel_VAE')
        voxel_vae.load_weights(weights_path, by_name=True)

        hash = os.listdir(voxel_data_path)
        voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]
        voxels = data_IO.voxelPathList2matrix(voxel_file_list)
        reconstructions = voxel_vae.predict(voxels)

    elif input_form == 'image':
        test_result_path = args.save_dir + '/test_sub_image_input'

        image_input = Input(shape=g.VIEWS_IMAGE_SHAPE)
        image_encoder = model.get_img_encoder(z_dim)
        decoder = model.get_voxel_decoder(z_dim)
        output = decoder(image_encoder(image_input))
        image_vae = Model(image_input, output, name='Test_Image_MVCNN_VAE')
        image_vae.load_weights(weights_path, by_name=True)

        hash = os.listdir(image_data_path)
        image_file_list = [os.path.join(image_data_path, id) for id in hash]
        voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]
        voxels = data_IO.voxelPathList2matrix(voxel_file_list)
        images = data_IO.imagePathList2matrix(image_file_list, train=False)
        reconstructions = image_vae.predict(images)

    elif input_form == 'both':
        test_result_path = args.save_dir + '/test_sub_both_input'

        mmi_vae = get_MMI(z_dim, 'weighted_add')
        mmi_vae.load_weights(weights_path, by_name=True)

        hash = os.listdir(image_data_path)
        image_file_list = [os.path.join(image_data_path, id) for id in hash]
        voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]

        images = data_IO.imagePathList2matrix(image_file_list, train=False)
        voxels = data_IO.voxelPathList2matrix(voxel_file_list)
        reconstructions = mmi_vae.predict([images, ])

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    # save the original test dataset file and generate the image
    if save_the_ori:
        voxel_path = voxel_data_path[:-9]
        ori_files_path = os.path.join(voxel_path, 'test_sub_visulization')
        ori_files = os.listdir(ori_files_path)
        for file in ori_files:
            file = os.path.join(ori_files_path, file)
            shutil.copy2(file, test_result_path)

    # save the generated objects files
    save_volume.save_metrics(reconstructions,voxels,voxel_data_path,image_data_path,input_form,test_result_path)
    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output_2(reconstructions[i, 0, :], hash[i], test_result_path, '_gen', save_bin=True,
                                         save_img=save_the_img)


if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))
