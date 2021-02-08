
import numpy as np
import os, shutil, random
from utils import data_IO
import tensorflow as tf
import scipy.ndimage as nd

def prepocess_dataset(voxel_dataset_path, image_dataset_path, split_scale=(0.8, 0.2)):

    object_number = len(os.listdir(voxel_dataset_path))
    train_dataset_size = int(object_number * split_scale[0])
    random.shuffle(voxel_dataset_path)

    voxel_train_dataset = voxel_dataset_path[:train_dataset_size]

def create_image_test_sub(voxel_test_sub_path, image_dataset, save_path):

    voxels_name = os.listdir(voxel_test_sub_path)
    voxels_hash = []
    for ele in voxels_name:
        h1 = ele.split('.')[0]
        voxels_hash.append(h1)

    for hash in voxels_hash:
        image_file = os.path.join(image_dataset, hash)
        save_to = os.path.join(save_path, hash)
        shutil.copytree(image_file, save_to)

def map_batch(img_batch, model_batch):
    data = []
    for each in range(img_batch.get_shape().as_list()[0]):
        img = decode_img(img_batch[each], [127, 127, 3])
        data.append(img)
    data = tf.stack(data, 0)
    return data, model_batch

def decode_img(path_tensor, img_size, channels = 3):
    tf.random.set_random_seed(2333)
    img = tf.read_file(path_tensor)
    img = tf.image.decode_image(img, channels)
    img = tf.image.random_crop(img, img_size)
    return img

if __name__ == '__main__':

    create_image_test_sub('/home/zmy/GitHub/MMI-VAE/dataset/03001627_test_sub',
                          '/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627',
                          '/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627_test')


