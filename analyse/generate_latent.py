import numpy as np
import shutil, sys, os, pickle

sys.path.append("..")

from MMI import *
from VAE import *
from utils import save_volume, data_IO, arg_parser, model

from utils import model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

ConFig = tf.ConfigProto()
ConFig.gpu_options.allow_growth = True
session = tf.Session(config=ConFig)


def latent2dict(hash, mu, logvar, z):
    output = {}
    for i in range(len(hash)):
        output[hash[i] + '_mu'] = mu[i]
        output[hash[i] + '_logvar'] = logvar[i]
        output[hash[i] + '_z'] = z[i]
    return output


def main(args):
    weights_path = args.weights_file
    save_the_img = args.generate_img
    save_the_ori = args.save_ori
    voxel_data_path = args.voxel_data_dir
    image_data_path = args.image_data_dir
    input_form = args.input_form

    z_dim = args.latent_vector_size

    if input_form == 'voxel':
        reconstructions_save_path = args.save_dir + '/analyse_voxel_input'
        latent_save_path = args.save_dir + '/voxel_latent_dict'
        os.makedirs(latent_save_path)

        voxel_input = Input(shape=g.VOXEL_INPUT_SHAPE)
        voxel_encoder = model.get_voxel_encoder(z_dim)
        decoder = model.get_voxel_decoder(z_dim)
        output = decoder(voxel_encoder(voxel_input))

        test_model = Model(voxel_input, output)

        test_model.load_weights(weights_path, by_name=True)
        voxel_encoder.load_weights(weights_path, by_name=True)
        decoder.load_weights(weights_path, by_name=True)

        hash = os.listdir(voxel_data_path)
        voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]
        voxels = data_IO.voxelPathList2matrix(voxel_file_list)

        # Get latent vector information
        z_mean, z_logvar, z = voxel_encoder.predict(voxels)
        epsilon = (z - z_mean) / np.exp(z_logvar)
        print("The epsilon in sampling layer is", epsilon)

        # record latent vectors in dictionary and save it in .pkl form
        latent_dict = latent2dict(hash, z_mean, z_logvar, z)
        latent_dict_save_path = os.path.join(latent_save_path, 'latent_dict.pkl')
        save_latent_dict = open(latent_dict_save_path, 'wb')
        pickle.dump(latent_dict, save_latent_dict)
        save_latent_dict.close()

        reconstructions = test_model.predict(voxels)

    elif input_form == 'image':
        reconstructions_save_path = args.save_dir + '/analyse_image_input'
        latent_save_path = args.save_dir + '/image_latent_dict'
        os.makedirs(latent_save_path)

        image_input = Input(shape=g.VIEWS_IMAGE_SHAPE)
        image_encoder = model.get_img_encoder(z_dim)
        image_encoder.load_weights(weights_path, by_name=True)

        decoder = model.get_voxel_decoder(z_dim)
        decoder.load_weights(weights_path, by_name=True)
        output = decoder(image_encoder(image_input)[0])
        test_model = Model(image_input, output)

        hash = os.listdir(image_data_path)
        image_file_list = [os.path.join(image_data_path, id) for id in hash]
        images = data_IO.imagePathList2matrix(image_file_list, train=False)

        # Get latent vector information
        z_mean, z_logvar, z = image_encoder.predict(images)
        print("The 1st z_mean is:", z_mean[0])
        print("The 10th z_mean is:", z_mean[9])
        print("The 20th z_mean is:", z_mean[19])
        print("The 30th z_mean is:", z_mean[29])
        print("The 40th z_mean is:", z_mean[39])
        print("The 1st latent vector is:", z[0])
        print("The 10th latent vector is:", z[9])
        print("The 20th latent vector is:", z[19])
        print("The 30th latent vector is:", z[29])
        print("The 40th latent vector is:", z[39])
        epsilon = (z - z_mean) / np.exp(z_logvar)
        #print("The epsilon in sampling layer is", epsilon)

        # record latent vectors in dictionary and save it in .pkl form
        latent_dict = latent2dict(hash, z_mean, z_logvar, z)
        latent_dict_save_path = os.path.join(latent_save_path, 'latent_dict.pkl')
        save_latent_dict = open(latent_dict_save_path, 'wb')
        pickle.dump(latent_dict, save_latent_dict)
        save_latent_dict.close()

        reconstructions = test_model.predict(images)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(reconstructions_save_path):
        os.makedirs(reconstructions_save_path)

    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output_2(reconstructions[i, 0, :], hash[i], reconstructions_save_path, '_gen',
                                         save_bin=True, save_img=save_the_img)


if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))
