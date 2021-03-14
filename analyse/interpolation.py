import numpy as np
import shutil, sys, os, pickle
sys.path.append("..")

from MMI import *
from VAE import *
from utils import save_volume, data_IO, arg_parser, model


ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)


def main():

    latent_dims = 128

    interpolation_save_path = '/home/zmy/Downloads/allCategory_uniLoss1/voxel_latent_dict/'
    latent_vector_file1 = open('/home/zmy/Downloads/allCategory_uniLoss1/voxel_latent_dict/latent_dict_airplane_test_sub.pkl', 'rb')
    latent_vector_file2 = open('/home/zmy/Downloads/allCategory_uniLoss1/voxel_latent_dict/latent_dict_car_test_sub.pkl', 'rb')
    weights_path = '/home/zmy/Downloads/allCategory_uniLoss1/weightsEnd_voxDecoder.h5'

    object1 = '4b4fd540cab0cdf3f38bce64a8733419'
    object2 = '117c0e0aafc0c3f81015cdff13e6d9f3'


    # Get the latent vector of two objects
    latent_vector_dict1 = pickle.load(latent_vector_file1)
    latent_vector_dict2 = pickle.load(latent_vector_file2)
    p1, p2 = latent_vector_dict1[object1+'_z'], latent_vector_dict2[object2+'_z']
    latent_vectors = np.linspace(p1,p2,11)



    # Define the decoder model
    decoder = model.get_voxel_decoder(latent_dims)
    decoder.load_weights(weights_path,by_name=True)
    reconstructions = decoder.predict(latent_vectors)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(interpolation_save_path):
        os.makedirs(interpolation_save_path)

    shutil.copy2(__file__, interpolation_save_path)
    for i in range(reconstructions.shape[0]):
        name = str(i)
        save_volume.save_binvox_output(reconstructions[i, 0, :], name, interpolation_save_path, '_gen', save_bin= True, save_img= True)

if __name__ == '__main__':
    main()