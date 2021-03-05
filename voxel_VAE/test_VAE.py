import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import shutil
import sys

from VAE import *
from utils import save_volume, data_IO, arg_parser

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def main(args):

    z_dim = args.latent_vector_size

    test_result_path = args.save_dir + '/test'
    voxel_data_path = args.voxel_data_dir

    model = get_voxel_VAE(z_dim)
    vae = model['vae']
    vae.load_weights(args.weights_file)

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
        up_level_path = os.path.split(voxel_data_path)[0]
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