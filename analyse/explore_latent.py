import numpy as np
import shutil, sys, os, pickle
sys.path.append("..")
from utils import save_volume, data_IO, arg_parser, model
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)


def main():

    latent_dims = 128

    interpolation_save_path = '/home/zmy/Downloads/bothTrain_lessFC_uniLoss1/voxel_latent_dict/latent_explore'
    if not os.path.exists(interpolation_save_path):
        os.makedirs(interpolation_save_path)
    shutil.copy2(__file__, interpolation_save_path)

    latent_vector_file1 = open('/home/zmy/Downloads/bothTrain_lessFC_uniLoss1/voxel_latent_dict/voxel_latent_dict_chair_all.pkl', 'rb')
    weights_path = '/home/zmy/Downloads/bothTrain_lessFC_uniLoss1/weightsEnd_voxDecoder.h5'

    object1 = '1a6f615e8b1b5ae4dbbc9440457e303e'
    latent_vector_dict1 = pickle.load(latent_vector_file1)

    # Print every dimension of latent vector
    p1 = latent_vector_dict1[object1+'_z_mean']

    for i_th_dim in range(latent_dims):

        i_interpolation_save_path = os.path.join(interpolation_save_path,'%d_th_dim'%i_th_dim)

        one_dim_changing_latents = np.tile(p1, (11,1))
        one_dim_changing_latents[:, i_th_dim] = np.linspace(-5.0,5.0,11)

        # Define the decoder model
        decoder = model.get_voxel_decoder(latent_dims)
        decoder.load_weights(weights_path,by_name=True)
        reconstructions = decoder.predict(one_dim_changing_latents)

        reconstructions[reconstructions > 0] = 1
        reconstructions[reconstructions < 0] = 0

        if not os.path.exists(i_interpolation_save_path):
            os.makedirs(i_interpolation_save_path)

        for i in range(reconstructions.shape[0]):
            name = str(i)
            save_volume.save_binvox_output(reconstructions[i, 0, :], name, i_interpolation_save_path, '_gen', save_bin= True, save_img= True)

if __name__ == '__main__':
    main()