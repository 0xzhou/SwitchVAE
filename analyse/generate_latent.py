

import numpy as np
import shutil, sys, os, pickle
sys.path.append("..")

from MMI import *
from VAE import *
from utils import save_volume, data_IO, arg_parser
from numpy import savez_compressed

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def latent2dict(hash, mu, sigma, z):
    output = {}
    for i in range(len(hash)):
        output[hash[i]+'_mu'] = mu[i]
        output[hash[i] + '_sigma'] = sigma[i]
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

    # Create new model that has only one input form and load specific part of trained weights
    if input_form == 'voxel':
        reconstructions_save_path = args.save_dir + '/analyse_voxel_input'
        latent_save_path = args.save_dir + '/voxel_latent_dict'
        os.makedirs(latent_save_path)

        model = get_voxel_VAE(z_dim)
        vae = model['vae']
        vae.load_weights(weights_path, by_name=True)

        voxel_data, hash = data_IO.voxelpath2matrix(voxel_data_path)
        encoder = model['encoder']
        mu, sigma, z = encoder.predict(voxel_data)

        # record latent vectors in dictionary and save it in .pkl form
        latent_dict = latent2dict(hash, mu, sigma, z)
        latent_dict_save_path = os.path.join(latent_save_path, 'latent_dict.pkl')
        save_latent_dict = open(latent_dict_save_path, 'wb')
        pickle.dump(latent_dict, save_latent_dict)
        save_latent_dict.close()

        reconstructions = vae.predict(voxel_data)

    elif input_form == 'image':
        reconstructions_save_path = args.save_dir + '/analyse_image_input'
        latent_save_path = args.save_dir + '/image_latent_dict'
        os.makedirs(latent_save_path)

        image_input = Input(shape=g.VIEWS_IMAGE_SHAPE)
        image_encoder = get_img_encoder(z_dim)
        decoder = get_voxel_decoder(z_dim)
        output = decoder(image_encoder(image_input))
        model = Model(image_input, output, name='Image_MVCNN_VAE')
        model.load_weights(weights_path, by_name=True)

        num_objects = len(os.listdir(image_data_path))
        images = np.zeros((num_objects,) + g.VIEWS_IMAGE_SHAPE, dtype=np.float32)
        object_files = os.listdir(image_data_path)
        hash = object_files

        for i, object in enumerate(object_files):
            image_path = os.path.join(image_data_path, object)
            images[i] = data_IO.imagepath2matrix(image_path)

        # record latent vectors in dictionary and save it in .pkl form
        mu, sigma, z = image_encoder.predict(images)
        latent_dict = latent2dict(hash, mu, sigma, z)
        latent_dict_save_path = os.path.join(latent_save_path, 'latent_dict.pkl')
        save_latent_dict = open(latent_dict_save_path, 'wb')
        pickle.dump(latent_dict, save_latent_dict)
        save_latent_dict.close()

        reconstructions = model.predict(images)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(reconstructions_save_path):
        os.makedirs(reconstructions_save_path)

    # save the original test dataset file and generate the image
    if save_the_ori:
        voxel_path = voxel_data_path[:-9]
        ori_files_path = os.path.join(voxel_path, 'test_sub_visulization')
        ori_files = os.listdir(ori_files_path)
        for file in ori_files:
            file = os.path.join(ori_files_path, file)
            shutil.copy2(file, reconstructions_save_path)

    # save the generated objects files
    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output(reconstructions[i, 0, :], hash[i], reconstructions_save_path, '_gen', save_bin= True, save_img= save_the_img)

if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))