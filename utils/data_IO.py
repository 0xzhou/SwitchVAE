
import numpy as np
import glob, os
import scipy.ndimage as nd
from utils import binvox_rw
from utils import globals as g

def read_voxel_data(model_path):
    with open(model_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        return model.data

def voxeldataset2matrix(voxel_dataset_path, padding = False):
    '''
    Transform the special dataset into arrays, in special dataset, objects are in the 'hash_id.binvox' form
    '''

    voxels_path = glob.glob(voxel_dataset_path + '/*')
    voxels_name = os.listdir(voxel_dataset_path)
    voxels_hash = []
    for ele in voxels_name:
        h1 = ele.split('.')[0]
        voxels_hash.append(h1)

    voxels = np.zeros((len(voxels_path),) + (1,32,32,32), dtype=np.float32)
    for i, name in enumerate(voxels_path):
        model = read_voxel_data(name)
        if padding:
            model = nd.zoom(model, (0.75, 0.75, 0.75), mode = 'constant', order = 0)
            model = np.pad(model, ((4,4),(4,4),(4,4)), 'constant')
        voxels[i] = model.astype(np.float32)
    return 3.0 * voxels -1.0, voxels_hash

def write_binvox_file(pred, filename):
    with open(filename, 'w') as f:
        voxel = binvox_rw.Voxels(pred, [32, 32, 32], [0, 0, 0], 1, 'xzy')
        #voxel = binvox_rw.Voxels(pred, [32, 32, 32], [0, 0, 0], 1, 'xyz')
        binvox_rw.write(voxel, f)
        f.close()

def imagepath2matrix(image_dataset_path, single_image_shape=(137, 137, 3) ):

    image_files = glob.glob(image_dataset_path + "/*/*" + "png")
    object_hash = os.listdir(image_dataset_path)

    images = np.zeros((24,) + single_image_shape, dtype=np.float32)
    for i, image in enumerate(image_files):
        images[i]= nd.imread(image,mode='RGB')
    return images

def voxelpath2matrix(voxel_dataset_path, padding = False):
    voxel_files = glob.glob(voxel_dataset_path+'/*')
    voxel_hash = os.listdir(voxel_dataset_path)

    num_objects = len(os.listdir(voxel_dataset_path))
    voxels = np.zeros((num_objects,)+g.VOXEL_INPUT_SHAPE, dtype=np.float32)
    for i, name in enumerate(voxel_files):
        name = glob.glob(name +'/*binvox')
        model = read_voxel_data(name[0])
        if padding:
            model = nd.zoom(model, (0.75, 0.75, 0.75), mode='constant', order=0)
            model = np.pad(model, ((4, 4), (4, 4), (4, 4)), 'constant')
        voxels[i] = model.astype(np.float32)
    return 3.0 * voxels - 1, voxel_hash


def generate_MMI_batch_data(voxel_path, image_path, batch_size):

    number_of_elements = len(os.listdir(voxel_path))
    hash_id = os.listdir(voxel_path)

    voxel_file_path = [os.path.join(voxel_path,id) for id in hash_id]
    image_file_path = [os.path.join(image_path,id) for id in hash_id]

    while 1:
        for start_idx in range(number_of_elements-batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)

            image_one_batch_files = image_file_path[excerpt]
            images_one_batch = np.zeros((batch_size,24) + g.IMAGE_SHAPE, dtype=np.float32)
            for i, element in enumerate(image_one_batch_files):
                images_one_batch[i] = imagepath2matrix(element)

            voxel_one_batch_files = voxel_file_path[excerpt]
            voxel_one_batch = np.zeros((batch_size,) + g.VOXEL_INPUT_SHAPE, dtype=np.float32)
            for i, element in enumerate(voxel_one_batch_files):
                model = glob.glob(element+'/*')
                model = read_voxel_data(model)
                voxel_one_batch[i] = model.astype(np.float32)
                break
            voxel_one_batch = 3.0 * voxel_one_batch - 1.0

            yield [images_one_batch, voxel_one_batch]




