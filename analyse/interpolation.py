
import numpy as np
import pickle, sys, os
sys.path.append("..")
from utils import save_volume
from VAE import *
from MMI import *
from utils.model import get_voxel_decoder


ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def interpolate_points(p1, p2, n_steps = 10):
    '''
    uniform interpolation between two points in latent space
    '''
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


def main():

    latent_dimensions = 128

    latent_vector_file = open('/home/zmy/Downloads/OneDrive-2021-02-15/voxel_latent_dict/latent_dict.pkl', 'rb')
    weights_path = '/home/zmy/Downloads/OneDrive-2021-02-15/weights_200_-6.2687.h5'
    interpolation_save_path = '/home/zmy/Downloads/OneDrive-2021-02-15/interpolation'
    if not os.path.exists(interpolation_save_path):
        os.makedirs(interpolation_save_path)

    latent_vector_dict = pickle.load(latent_vector_file)
    hash_id_1 = '117c0e0aafc0c3f81015cdff13e6d9f3'
    hash_id_2 = '8951c681ee693af213493f4cb10e07b0'
    latent_vector_1 = latent_vector_dict[hash_id_1+'_z']
    latent_vector_2 = latent_vector_dict[hash_id_2+'_z']
    print("latent_vector_1 is", latent_vector_1)
    print("latent_vector_2 is", latent_vector_2)

    interpolated_vectors = interpolate_points(latent_vector_1, latent_vector_2, 11)
    print("The shape of interpolated_vectors", interpolated_vectors.shape)
    print("The first element of interpolated_vectors", interpolated_vectors[0])
    print("The type of first element of interpolated_vectors", type(interpolated_vectors[0]))
    voxel_decoder = get_voxel_decoder(latent_dimensions)
    voxel_decoder.load_weights(weights_path, by_name=True)

    # vae = get_voxel_VAE(latent_dimensions)['vae']
    # vae.load_weights(weights_path)
    # voxel_generator = vae.get_layer(name='Voxel_Generator')
    #interpolated_vectors = interpolated_vectors.astype(np.float32)
    reconstructions = voxel_decoder.predict(interpolated_vectors)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    # save the generated objects files
    for i in range(reconstructions.shape[0]):
        name = str(i)
        save_volume.save_binvox_output(reconstructions[i, 0, :], name, interpolation_save_path, '',
                                       save_bin=True, save_img=True)

if __name__ == '__main__':
    main()




