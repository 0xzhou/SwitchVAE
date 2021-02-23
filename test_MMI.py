import os
import numpy as np
import tensorflow as tf
import shutil, sys

from tensorflow.keras.layers import Input, Lambda, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

from utils import save_volume, data_IO, arg_parser, model
from utils import globals as g
from MMI import *

ConFig = tf.ConfigProto()
ConFig.gpu_options.allow_growth = True
session = tf.Session(config=ConFig)


def main(args):
    weights_path = args.weights_file
    save_the_img = args.generate_img
    save_the_ori = args.save_ori
    voxel_data_path = args.voxel_data_dir
    image_data_path = args.image_data_dir
    input_form = args.input_form
    z_dim = args.latent_vector_size

    model_pdf_path = os.path.join(args.save_dir,'model_pdf_test')
    os.makedirs(model_pdf_path)

    if input_form == 'voxel':
        test_result_path = args.save_dir + '/test_sub_voxel_input'

        voxel_input = Input(shape=g.VOXEL_INPUT_SHAPE)
        voxel_encoder = model.get_voxel_encoder(z_dim)
        voxel_encoder.load_weights(weights_path, by_name=True)
        decoder = model.get_voxel_decoder(z_dim)
        decoder.load_weights(weights_path, by_name=True)
        output = decoder(voxel_encoder(voxel_input)[0])
        voxel_vae = Model(voxel_input, output, name='Test_Voxel_VAE')
        voxel_vae.load_weights(weights_path, by_name=True)

        hash = os.listdir(voxel_data_path)
        voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]
        voxels = data_IO.voxelPathList2matrix(voxel_file_list)
        reconstructions = voxel_vae.predict(voxels)

        plot_model(voxel_encoder, to_file=os.path.join(model_pdf_path,'Voxel_Encoder.pdf'), show_shapes=True)
        plot_model(decoder, to_file=os.path.join(model_pdf_path,'Encoder.pdf'), show_shapes=True)
        plot_model(voxel_vae, to_file=os.path.join(model_pdf_path, 'Voxel_VAE.pdf'), show_shapes=True)


    elif input_form == 'image':
        test_result_path = args.save_dir + '/test_sub_image_input'

        image_input = Input(shape=g.VIEWS_IMAGE_SHAPE)
        image_encoder = model.get_img_encoder(z_dim)
        image_encoder.load_weights(weights_path, by_name=True)

        decoder = model.get_voxel_decoder(z_dim)
        decoder.load_weights(weights_path, by_name=True)
        output = decoder(image_encoder(image_input)[0])
        image_vae = Model(image_input, output)

        hash = os.listdir(image_data_path)
        image_file_list = [os.path.join(image_data_path, id) for id in hash]
        voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]
        voxels = data_IO.voxelPathList2matrix(voxel_file_list)
        images = data_IO.imagePathList2matrix(image_file_list, train=False)
        reconstructions = image_vae.predict(images)

        plot_model(image_encoder, to_file=os.path.join(model_pdf_path,'Image_Encoder.pdf'), show_shapes=True)
        plot_model(decoder, to_file=os.path.join(model_pdf_path, 'Decoder.pdf'), show_shapes=True)
        plot_model(image_vae, to_file=os.path.join(model_pdf_path, 'Image_VAE.pdf'), show_shapes=True)

    elif input_form == 'both':
        test_result_path = args.save_dir + '/test_sub_both_input'

        image_input = Input(shape=g.VIEWS_IMAGE_SHAPE)
        voxel_input = Input(shape=g.VOXEL_INPUT_SHAPE)
        image_encoder = model.get_img_encoder(z_dim)
        voxel_encoder = model.get_voxel_encoder(z_dim)
        image_encoder.load_weights(weights_path, by_name=True)
        voxel_encoder.load_weights(weights_path, by_name=True)

        img_encoder_output = image_encoder(image_input)
        vol_encoder_output = voxel_encoder(voxel_input)
        weight_op_img = Lambda(lambda x: x * g.IMG_WEIGHT, name='Imgae_Weighted_Layer')
        weight_op_vol = Lambda(lambda x: x * g.VOL_WEIGHT, name='Voxel_Weighted_Layer')
        img_z_mean, img_z_logvar, img_z = [weight_op_img(x) for x in img_encoder_output]
        vol_z_mean, vol_z_logvar, vol_z = [weight_op_vol(x) for x in vol_encoder_output]
        z_mean = Add(name='Weighted_Add_z_mean')([img_z_mean, vol_z_mean])
        z_logvar = Add(name='Weighted_Add_z_logvar')([img_z_logvar, vol_z_logvar])
        z = Add(name='Weighted_Add_z')([img_z, vol_z])

        decoder = model.get_voxel_decoder(z_dim)
        decoder.load_weights(weights_path, by_name=True)
        output = decoder(z_mean)

        mmi = Model([image_input,voxel_input],output)

        hash = os.listdir(image_data_path)
        image_file_list = [os.path.join(image_data_path, id) for id in hash]
        voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]

        images = data_IO.imagePathList2matrix(image_file_list, train=False)
        voxels = data_IO.voxelPathList2matrix(voxel_file_list)
        reconstructions = mmi.predict([images, voxels])

        plot_model(image_encoder, to_file=os.path.join(model_pdf_path,'Image_Encoder.pdf'), show_shapes=True)
        plot_model(image_encoder, to_file=os.path.join(model_pdf_path, 'Voxel_Encoder.pdf'), show_shapes=True)
        plot_model(decoder, to_file=os.path.join(model_pdf_path, 'Decoder.pdf'), show_shapes=True)
        plot_model(mmi, to_file=os.path.join(model_pdf_path, 'MMI.pdf'), show_shapes=True)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    # save the original test dataset file and generate the image
    if save_the_ori:
        up_level_path = os.path.split(voxel_data_path)[0]
        ori_files_path = os.path.join(up_level_path, 'test_sub_visulization')
        ori_files = os.listdir(ori_files_path)
        for file in ori_files:
            file = os.path.join(ori_files_path, file)
            shutil.copy2(file, test_result_path)

    # save the generated objects files
    save_volume.save_metrics(reconstructions,voxels,voxel_data_path,image_data_path,input_form,test_result_path)
    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output_2(reconstructions[i, 0, :], hash[i], test_result_path, '_gen', save_bin=True,
                                         save_img=save_the_img)


if __name__ == '__main__':
    main(arg_parser.parse_test_arguments(sys.argv[1:]))
