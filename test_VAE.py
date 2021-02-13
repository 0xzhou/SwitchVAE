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

    weights_path = args.weights_file
    test_result_path = args.save_dir + '/test/'
    save_the_img = args.generate_img
    save_the_ori = args.save_ori
    test_data_path = args.test_data_dir

    model = get_model(z_dim)

    inputs = model['inputs']
    outputs = model['outputs']
    mu = model['mu']
    sigma = model['sigma']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']
    vae = model['vae']

    # Set the weight files and test dataset path
    vae.load_weights(weights_path)
    data_test, hash = data_IO.voxelpath2matrix(test_data_path)

    reconstructions = vae.predict(data_test)
    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    # copy the original test dataset file
    if save_the_ori:
        for i in range(reconstructions.shape[0]):
            shutil.copy2('./dataset/03001627_test_sub/'+hash[i]+'.binvox', test_result_path)
            if save_the_img:
                shutil.copy2('./dataset/03001627_test_sub_visualization/' + hash[i] + '.png', test_result_path)

    # save the generated objects files
    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output(reconstructions[i, 0, :], hash[i], test_result_path, '_gen', save_bin= True, save_img= save_the_img)

if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))