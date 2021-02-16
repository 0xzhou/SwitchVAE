import numpy as np
import shutil, sys, os, pickle
sys.path.append("..")

from MMI import *
from VAE import *
from utils import save_volume, data_IO, arg_parser, model

from utils import model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)


def main():

    latent_dims = 128

    interpolation_save_path = '/home/zmy/Downloads/OneDrive-2021-02-15/interpolation'
    latent_vector_file = open('/home/zmy/Downloads/OneDrive-2021-02-15/voxel_latent_dict/latent_dict.pkl', 'rb')
    weights_path = '/home/zmy/Downloads/OneDrive-2021-02-15/weights_200_-6.2687.h5'

    # Define the decoder model
    voxel_input = Input(shape=g.VOXEL_INPUT_SHAPE)
    voxel_encoder = model.get_voxel_encoder_old(latent_dims)
    decoder = model.get_voxel_decoder_old(latent_dims)
    output = decoder(voxel_encoder(voxel_input))
    test_model = Model(voxel_input, output)
    test_model.load_weights(weights_path, by_name=True)
    voxel_encoder.load_weights(weights_path,by_name= True)
    decoder.load_weights(weights_path,by_name=True)

    # Get the latent vector of two objects
    latent_vector_dict1 = pickle.load(latent_vector_file)
    p1, p2 = latent_vector_dict1['5c86904bdc50a1ca173c8feb9cba831_z'], latent_vector_dict1['5cc0b0e0035170434733824eae5cd9ae_z']

    latent_vectors = np.linspace(p1, p2, 11)
    reconstructions = decoder.predict(latent_vectors)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(interpolation_save_path):
        os.makedirs(interpolation_save_path)

    for i in range(reconstructions.shape[0]):
        name = str(i)
        save_volume.save_binvox_output_2(reconstructions[i, 0, :], name, interpolation_save_path, '_gen', save_bin= True, save_img= True)

if __name__ == '__main__':
    main()