
import numpy as np
import os, shutil
import glob
from utils import npytar, binvox_IO
import tensorflow as tf
import scipy.ndimage as nd

# def preprocess_shapenet_rendering(rendering_path, output_path):
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#
#     id_hash_set = os.listdir(rendering_path)
#     for id in id_hash_set:
#         img_path = os.path.join(rendering_path, id, 'rendering')
#         imgs = os.listdir(img_path)
#         for img in imgs:
#             if img.endswith('.png'):
#                 new_img_name = id + img
#                 shutil.copy2(img, os.path.join(output_path, new_img_name))
#         break
#
#
#     print(id_hash_set[:10])

def imagepath2matrix(id_path, image_shape=(137,137,3)):
    image_files = glob.glob(id_path + "/*/*" + "png")
    channels = image_shape[2]
    images = np.zeros((24,)+image_shape, dtype=np.float32)
    for i, image in enumerate(image_files):
        images[i]= nd.imread(image,mode='RGB')
    return images


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



# if __name__ == '__main__':
#     voxel_dataset_path='/home/zmy/GitHub/MMI-VAE/dataset/03001627_train'
#     voxel_data_train, hash = binvox_IO.voxelpath2matrix(voxel_dataset_path)  # Number of element * 1 * 32 * 32 * 32
#     rendering_dic = '/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627'
#     id_path = '/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627/1a6f615e8b1b5ae4dbbc9440457e303e'
#     #preprocess_shapenet_rendering(rendering_dic, './03001627_img')
#     #imagepath2matrix(rendering_dic)
#     #image_data_path = glob.glob(rendering_dic+"/*")
#     imagepath2matrix(id_path)


