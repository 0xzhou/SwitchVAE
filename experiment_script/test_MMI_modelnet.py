import os
import numpy as np
import tensorflow as tf
import shutil, sys, random

from tensorflow.keras.layers import Input, Lambda, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

from utils import save_volume, data_IO, arg_parser, model
from utils import globals as g
from SwitchVAE import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ConFig = tf.ConfigProto()
ConFig.gpu_options.allow_growth = True
session = tf.Session(config=ConFig)


def main(args):
    weights_dir = args.weights_dir
    save_the_img = args.generate_img
    save_bin = args.save_bin
    save_the_ori = args.save_ori
    modelnet_voxel_dataset = args.modelnet_voxel_dataset
    modelnet_image_dataset = args.modelnet_image_dataset
    input_form = args.input_form
    z_dim = args.latent_vector_size
    batch_size = args.batch_size

    ModelNet10_CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                          'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    ModelNet40_CLASSES = [ 'airplane', 'bowl', 'table', 'chair', 'vase', 'glass_box', 'bathtub', 'toilet', 'range_hood',
                          'flower_pot', 'laptop', 'plant', 'cup', 'person', 'tent', 'sofa', 'monitor', 'keyboard',
                          'desk', 'mantel', 'curtain', 'bed', 'lamp', 'bench', 'dresser','car', 'sink',
                          'night_stand', 'stool', 'door', 'guitar', 'stairs', 'radio', 'tv_stand', 'cone', 'xbox',
                          'wardrobe', 'bookshelf', 'bottle', 'piano']

    multi_category_id = data_IO.generate_modelnet_idList(modelnet_voxel_dataset, ModelNet40_CLASSES, 'test')

    random.shuffle(multi_category_id)
    num_test_object = len(multi_category_id)

    model_pdf_path = os.path.join(args.save_dir, 'model_modelnet_pdf_test')
    if not os.path.exists(model_pdf_path):
        os.makedirs(model_pdf_path)

    if input_form == 'voxel':
        test_result_path = args.save_dir + '/test_modelnet_voxel_input'

        voxel_input = Input(shape=g.VOXEL_INPUT_SHAPE)
        voxel_encoder = model.get_voxel_encoder(z_dim)
        voxel_encoder.load_weights(os.path.join(weights_dir, 'weightsEnd_voxEncoder.h5'), by_name=True)
        decoder = model.get_voxel_decoder(z_dim)
        decoder.load_weights(os.path.join(weights_dir, 'weightsEnd_voxDecoder.h5'), by_name=True)
        output = decoder(voxel_encoder(voxel_input)[0])
        voxel_vae = Model(voxel_input, output, name='Test_Voxel_VAE')

        voxels = np.zeros((num_test_object,) + g.VOXEL_INPUT_SHAPE, dtype=np.float32)
        for i , cat_id in enumerate(multi_category_id):
            category, hash = cat_id.rsplit('_', 1)[0], cat_id.rsplit('_', 1)[1]
            voxel_file = os.path.join(modelnet_voxel_dataset, category, 'test', cat_id + '.binvox')
            voxels[i] = data_IO.read_voxel_data(voxel_file)

        reconstructions = voxel_vae.predict(voxels)

        plot_model(voxel_encoder, to_file=os.path.join(model_pdf_path, 'Voxel_Encoder.pdf'), show_shapes=True)
        plot_model(decoder, to_file=os.path.join(model_pdf_path, 'Encoder.pdf'), show_shapes=True)
        plot_model(voxel_vae, to_file=os.path.join(model_pdf_path, 'Voxel_VAE.pdf'), show_shapes=True)

    elif input_form == 'image':
        test_result_path = args.save_dir + '/test_modelnet_image_input'
        image_input = Input(shape=g.VIEWS_IMAGE_SHAPE_MODELNET)
        image_encoder = model.get_img_encoder(z_dim, g.VIEWS_IMAGE_SHAPE_MODELNET)['image_encoder']
        image_encoder.load_weights(os.path.join(weights_dir, 'weightsEnd_imgEncoder.h5'), by_name=True)
        decoder = model.get_voxel_decoder(z_dim)
        decoder.load_weights(os.path.join(weights_dir, 'weightsEnd_voxDecoder.h5'), by_name=True)
        output = decoder(image_encoder(image_input)[0])
        image_vae = Model(image_input, output)

        images = np.zeros((num_test_object, 12) + g.IMAGE_SHAPE, dtype=np.float32)
        voxels = np.zeros((num_test_object,) + g.VOXEL_INPUT_SHAPE, dtype=np.float32)
        for i, cat_id in enumerate(multi_category_id):
            category, hash = cat_id.rsplit('_', 1)[0], cat_id.rsplit('_', 1)[1]
            image_prefix = os.path.join(modelnet_image_dataset, category, 'test', cat_id)
            view_image = np.zeros(g.VIEWS_IMAGE_SHAPE_MODELNET, dtype=np.float32)
            for view in range(12):
                image_file = image_prefix + '.obj.shaded_v' + str(view + 1).zfill(3) + '.png'
                image = data_IO.preprocess_modelnet_img(image_file)
                view_image[view] = image
            images[i] = view_image

            voxel_file = os.path.join(modelnet_voxel_dataset, category, 'test', cat_id + '.binvox')
            voxels[i] = data_IO.read_voxel_data(voxel_file)
            print("Loading image from dataset:",str(i) + '/'+ str(num_test_object))

        reconstructions = image_vae.predict(images)

        plot_model(image_encoder, to_file=os.path.join(model_pdf_path, 'Image_Encoder.pdf'), show_shapes=True)
        plot_model(decoder, to_file=os.path.join(model_pdf_path, 'Decoder.pdf'), show_shapes=True)
        plot_model(image_vae, to_file=os.path.join(model_pdf_path, 'Image_VAE.pdf'), show_shapes=True)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    # save the original test dataset file and generate the image
    if save_the_ori:
        pass

    # save the generated objects files
    save_volume.save_metrics(reconstructions, voxels, modelnet_voxel_dataset, modelnet_image_dataset, input_form, test_result_path)

    #for i in range(reconstructions.shape[0]):
    if save_bin or save_the_img:
        for i in range(100):
            save_volume.save_binvox_output_for_modelnet(reconstructions[i, 0, :], multi_category_id[i], test_result_path, '_gen', save_bin=save_bin,
                                       save_img=save_the_img)

if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))
