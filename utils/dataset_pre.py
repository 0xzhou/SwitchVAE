
import numpy as np
import os, shutil, random
import tensorflow as tf
import scipy.ndimage as nd

def create_dataset(voxel_dataset_path, image_dataset_path, split_scale=(0.8, 0.2), save_path=None):
    '''
    Args:
        split_scale: (number of train samples, number of test samples)
        save_path: function will create subfolders under the save_path

    Returns: 0
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    object_number = len(os.listdir(voxel_dataset_path))
    train_dataset_size = int(object_number * split_scale[0])
    voxel_file_names = os.listdir(voxel_dataset_path)
    random.shuffle(voxel_file_names)

    voxel_train_names = voxel_file_names[:train_dataset_size]
    voxel_test_names = voxel_file_names[train_dataset_size:]
    vol_train_dataset_save_path = os.path.join(save_path, 'voxel_trainset')
    vol_test_dataset_save_path = os.path.join(save_path, 'voxel_testset')
    img_train_dataset_save_path = os.path.join(save_path, 'image_trainset')
    img_test_dataset_save_path = os.path.join(save_path, 'image_testset')

    for file in voxel_train_names:
        vol_original_path = os.path.join(voxel_dataset_path, file)
        img_original_path = os.path.join(image_dataset_path, file)
        vol_save_to = os.path.join(vol_train_dataset_save_path, file)
        img_save_to = os.path.join(img_train_dataset_save_path, file)
        shutil.copytree(vol_original_path, vol_save_to)
        shutil.copytree(img_original_path, img_save_to)
    for file in voxel_test_names:
        vol_original_path = os.path.join(voxel_dataset_path, file)
        img_original_path = os.path.join(image_dataset_path, file)
        vol_save_to = os.path.join(vol_test_dataset_save_path, file)
        img_save_to = os.path.join(img_test_dataset_save_path, file)
        shutil.copytree(vol_original_path, vol_save_to)
        shutil.copytree(img_original_path, img_save_to)

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
    create_dataset(
        voxel_dataset_path='/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetVox32/04256520',
        image_dataset_path='/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/04256520',
        save_path='/home/zmy/Datasets/3d-r2n2-datasat/04256520_new'
    )




