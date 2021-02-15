
import numpy as np
import pickle, sys, os
sys.path.append("..")
from utils import model, save_volume
from VAE import *

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

    latent_vector_file = open('/home/zmy/TrainingData/training3/2021_02_14_12_03_52/image_latent_dict/latent_dict.pkl', 'rb')
    weights_path = '/home/zmy/TrainingData/tf1.x.keras/2021_02_02_11_51_46/weights.h5'
    interpolation_save_path = '/home/zmy/TrainingData/training3/2021_02_14_12_03_52/interpolation'
    os.makedirs(interpolation_save_path)

    latent_vector_dict = pickle.load(latent_vector_file)
    hash_id_1 = '3b513237d90a4cd1576d8983ea1341c3'
    hash_id_2 = '3e53710a62660c60c39d538df4c93695'
    latent_vector_1 = latent_vector_dict[hash_id_1+'_z']
    latent_vector_2 = latent_vector_dict[hash_id_2+'_z']
    print("latent_vector_1 is", latent_vector_1)
    print("latent_vector_2 is", latent_vector_2)

    interpolated_vectors = interpolate_points(latent_vector_1, latent_vector_2, 10)

    vae = get_voxel_VAE(latent_dimensions)['vae']
    vae.load_weights(weights_path)
    voxel_generator = vae.get_layer(name='Voxel_Generator')

    reconstructions = voxel_generator.predict(interpolated_vectors)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    # save the generated objects files
    for i in range(reconstructions.shape[0]):
        name = str(i)
        save_volume.save_binvox_output(reconstructions[i, 0, :], name, interpolation_save_path, '',
                                       save_bin=True, save_img=True)

if __name__ == '__main__':
    main()




