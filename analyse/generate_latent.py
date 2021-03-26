import numpy as np
import shutil, sys, os, pickle

sys.path.append("..")

from MMI import *
from utils import save_volume, data_IO, arg_parser, model

from utils import model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.utils import shuffle
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ConFig = tf.ConfigProto()
ConFig.gpu_options.allow_growth = True
session = tf.Session(config=ConFig)


def latent2dict(hash, z_mean, z_logvar, z):
    output = {}
    for i in range(len(hash)):
        output[hash[i] + '_z_mean'] = z_mean[i]
        output[hash[i] + '_z_logvar'] = z_logvar[i]
        output[hash[i] + '_z'] = z[i]
    return output


def main(args):
    weights_dir = args.weights_dir
    save_the_img = args.generate_img
    voxel_data_path = args.voxel_data_dir
    image_data_path = args.image_data_dir
    input_form = args.input_form
    dataset = args.dataset

    z_dim = args.latent_vector_size

    if input_form == 'voxel':
        reconstructions_save_path = args.save_dir + '/analyse_voxel_input'
        latent_save_path = args.save_dir + '/voxel_latent_dict'
        if not os.path.exists(latent_save_path):
            os.makedirs(latent_save_path)

        voxel_input = Input(shape=g.VOXEL_INPUT_SHAPE)
        voxel_encoder = model.get_voxel_encoder(z_dim)
        decoder = model.get_voxel_decoder(z_dim)
        output = decoder(voxel_encoder(voxel_input))
        test_model = Model(voxel_input, output)

        voxel_encoder.load_weights(os.path.join(weights_dir, 'weightsEnd_voxEncoder.h5'), by_name=True)
        decoder.load_weights(os.path.join(weights_dir, 'weightsEnd_voxDecoder.h5'), by_name=True)

        if dataset == 'shapenet':
            hash = os.listdir(voxel_data_path)
            voxel_file_list = [os.path.join(voxel_data_path, id) for id in hash]
            voxels = data_IO.voxelPathList2matrix(voxel_file_list)

            z_mean, z_logvar, z = voxel_encoder.predict(voxels)
            latent_dict = latent2dict(hash, z_mean, z_logvar, z)
            latent_dict_save_path = os.path.join(latent_save_path, 'voxel_latent_dict_table_all.pkl')
            save_latent_dict = open(latent_dict_save_path, 'wb')
            pickle.dump(latent_dict, save_latent_dict)
            save_latent_dict.close()
            reconstructions = test_model.predict(voxels)

        elif dataset == 'modelnet':
            X = {'train_z_mean': [], 'train_z': [], 'test_z_mean':[], 'test_z':[],'train_z_cat':[] ,'test_z_cat':[]}
            y = {'train_label': [], 'test_label': []}
            voxel_data = np.load(args.voxel_npz)
            train_voxels, train_labels = shuffle(voxel_data['X_train'], voxel_data['y_train'])
            test_voxels, test_labels = shuffle(voxel_data['X_test'], voxel_data['y_test'])
            train_batch = int(train_voxels.shape[0] / args.batch_size)
            test_batch = int(test_voxels.shape[0] / args.batch_size)
            for i in range(train_batch):
                train_batch_voxels = train_voxels[args.batch_size * i:args.batch_size * (i + 1),:]
                train_batch_labels = train_labels[args.batch_size * i:args.batch_size * (i + 1)]
                train_z_mean, train_z_logvar, train_z = voxel_encoder.predict(train_batch_voxels)
                train_z_concatenate = np.concatenate((train_z_mean, train_z_logvar), 1)
                for j in range(train_z_mean.shape[0]):
                    X['train_z_mean'].append(train_z_mean[j])
                    X['train_z'].append(train_z[j])
                    X['train_z_cat'].append(train_z_concatenate[j])
                    y['train_label'].append(train_batch_labels[j])
            for i in range(test_batch):
                test_batch_voxels = test_voxels[args.batch_size * i:args.batch_size * (i + 1), :]
                test_batch_labels = test_labels[args.batch_size * i:args.batch_size * (i + 1)]
                test_z_mean, test_z_logvar, test_z = voxel_encoder.predict(test_batch_voxels)
                test_z_concatenate = np.concatenate((test_z_mean, test_z_logvar), 1)
                for j in range(test_z_mean.shape[0]):
                    X['test_z_mean'].append(test_z_mean[j])
                    X['test_z'].append(test_z[j])
                    X['test_z_cat'].append(test_z_concatenate[j])
                    y['test_label'].append(test_batch_labels[j])

            np.savez_compressed(os.path.join(args.save_dir,'modelnet10_voxel_latent_cat.npz'),
                                train_z=X['train_z'],
                                train_z_mean=X['train_z_mean'],
                                train_z_cat=X['train_z_cat'],
                                train_labels=y['train_label'],
                                test_z=X['test_z'],
                                test_z_mean=X['test_z_mean'],
                                test_z_cat=X['test_z_cat'],
                                test_labels=y['test_label'])

    elif input_form == 'image':
        reconstructions_save_path = args.save_dir + '/analyse_image_input'
        latent_save_path = args.save_dir + '/image_latent_dict'
        if not os.path.exists(latent_save_path):
            os.makedirs(latent_save_path)

        image_input = Input(shape=g.VIEWS_IMAGE_SHAPE_SHAPENET)
        image_encoder = model.get_img_encoder(z_dim)['image_encoder']
        image_encoder.load_weights(os.path.join(weights_dir, 'weightsEnd_imgEncoder.h5'), by_name=True)

        decoder = model.get_voxel_decoder(z_dim)
        decoder.load_weights(os.path.join(weights_dir, 'weightsEnd_voxDecoder.h5'), by_name=True)
        output = decoder(image_encoder(image_input)[0])
        test_model = Model(image_input, output)

        if dataset == 'shapenet':

            hash = os.listdir(image_data_path)
            image_file_list = [os.path.join(image_data_path, id) for id in hash]
            images = data_IO.imagePathList2matrix(image_file_list, train=False)

            # Get latent vector information
            z_mean, z_logvar, z = image_encoder.predict(images)

            # record latent vectors in dictionary and save it in .pkl form
            latent_dict = latent2dict(hash, z_mean, z_logvar, z)
            latent_dict_save_path = os.path.join(latent_save_path, 'latent_dict.pkl')
            save_latent_dict = open(latent_dict_save_path, 'wb')
            pickle.dump(latent_dict, save_latent_dict)
            save_latent_dict.close()

            reconstructions = test_model.predict(images)
        elif dataset == 'modelnet':
            X = {'train_z_mean': [], 'train_z': [], 'test_z_mean': [], 'test_z': []}
            y = {'train_label': [], 'test_label': []}
            object_id_data = np.load(args.image_npz)
            modelnet_image_path = '/home/zmy/Datasets/ModelNet40_images/modelnet40_images_new_12x'
            train_images_id, train_labels = shuffle(object_id_data['X_train'], object_id_data['y_train'])
            test_images_id, test_labels = shuffle(object_id_data['X_test'], object_id_data['y_test'])
            train_batch = int(train_images_id.shape[0] / args.batch_size)
            test_batch = int(test_images_id.shape[0] / args.batch_size)
            count=1

            for i in range(train_batch):
                train_batch_object_id = train_images_id[args.batch_size * i:args.batch_size * (i + 1)]
                train_batch_images = data_IO.objectIdList2matrix(train_batch_object_id,modelnet_image_path,'train')
                train_batch_labels = train_labels[args.batch_size * i:args.batch_size * (i + 1)]
                print("Predicting train batch:", str(count)+'/'+str(train_batch))
                count += 1
                train_z_mean, train_z_logvar, train_z = image_encoder.predict(train_batch_images)
                for j in range(train_z_mean.shape[0]):
                    X['train_z_mean'].append(train_z_mean[j])
                    X['train_z'].append(train_z[j])
                    y['train_label'].append(train_batch_labels[j])
            count =1
            for i in range(test_batch):
                test_batch_object_id = test_images_id[args.batch_size * i:args.batch_size * (i + 1)]
                test_batch_images = data_IO.objectIdList2matrix(test_batch_object_id, modelnet_image_path, 'test')
                test_batch_labels = test_labels[args.batch_size * i:args.batch_size * (i + 1)]
                print("Predicting test batch:", str(count) + '/' + str(test_batch))
                count += 1
                test_z_mean, test_z_logvar, test_z = image_encoder.predict(test_batch_images)
                for j in range(test_z_mean.shape[0]):
                    X['test_z_mean'].append(test_z_mean[j])
                    X['test_z'].append(test_z[j])
                    y['test_label'].append(test_batch_labels[j])

            np.savez_compressed(os.path.join(args.save_dir, 'modelnet10_image_BG255_latent_2.npz'),
                                train_z=X['train_z'],
                                train_z_mean=X['train_z_mean'],
                                train_labels=y['train_label'],
                                test_z=X['test_z'],
                                test_z_mean=X['test_z_mean'],
                                test_labels=y['test_label'])

    if bool(args.generation):

        reconstructions[reconstructions > 0] = 1
        reconstructions[reconstructions < 0] = 0

        if not os.path.exists(reconstructions_save_path):
            os.makedirs(reconstructions_save_path)

        save_volume.save_metrics(reconstructions, voxels, voxel_data_path, image_data_path, input_form, reconstructions_save_path)

        for i in range(reconstructions.shape[0]):
            save_volume.save_binvox_output_2(reconstructions[i, 0, :], hash[i], reconstructions_save_path, '_gen',
                                             save_bin=True, save_img=save_the_img)


if __name__ == '__main__':
    start = time.time()
    main(arg_parser.parse_test_arguments(sys.argv[1:]))
    end = time.time()
    interval = int(end) - int(start)
    print("The time spent on seconds: ", interval)