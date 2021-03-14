import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import shutil
import sys

from VAE import *
from utils import save_volume, data_IO, arg_parser, model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def main(args):

    z_dim = args.latent_vector_size

    test_result_path = args.save_dir + '/test'
    voxel_data_path = args.voxel_data_dir
    weights_dir = args.weights_dir

    # Load method 1
    # model = get_voxel_VAE(z_dim)
    # vae = model['vae']
    # vae.load_weights(os.path.join(weights_dir, 'weightsEnd_all.h5'))

    # Load method 2
    voxel_input = Input(shape=g.VOXEL_INPUT_SHAPE)
    voxel_encoder = model.get_voxel_encoder(z_dim)
    voxel_encoder.load_weights(os.path.join(weights_dir, 'weightsEnd_voxEncoder.h5'), by_name=True)

    decoder = model.get_voxel_decoder(z_dim)
    decoder.load_weights(os.path.join(weights_dir, 'weightsEnd_voxDecoder.h5'), by_name=True)

    output = decoder(voxel_encoder(voxel_input)[0])
    vae = Model(voxel_input, output, name='Test_Voxel_VAE')

    hash = os.listdir(voxel_data_path)
    voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]
    voxels = data_IO.voxelPathList2matrix(voxel_file_list)

    reconstructions = vae.predict(voxels)
    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    # copy the original test dataset file
    if bool(args.save_ori):
        up_level_path = voxel_data_path.rsplit('/',1)[0]
        print(up_level_path)
        ori_files_path = os.path.join(up_level_path, 'test_sub_visulization')
        ori_files = os.listdir(ori_files_path)
        for file in ori_files:
            file = os.path.join(ori_files_path, file)
            shutil.copy2(file, test_result_path)

    # save the generated objects files
    save_volume.save_metrics(reconstructions, voxels, voxel_data_path, '', 'voxel', test_result_path)
    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output(reconstructions[i, 0, :], hash[i], test_result_path, '_gen', save_bin= bool(args.save_bin), save_img= bool(args.generate_img))

if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))