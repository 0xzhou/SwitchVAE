
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

from MMI import *
from utils import data_IO, arg_parser, save_train, custom_loss, model
import sys, os, glob


ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

if __name__ == '__main__':

    voxel_decoder = model.get_voxel_decoder(200)
    plot_model(voxel_decoder, to_file='./voxel_decoder.pdf', show_shapes=True)
    image_decoder = model.get_img_encoder(200)
    plot_model(image_decoder, to_file='./image_decoder.pdf', show_shapes=True)

def voxeldataset2matrix(voxel_dataset_path, padding=False):
    '''
    Transform the special dataset into arrays, in special dataset, objects are in the 'hash_id.binvox' form
    '''

    voxels_path = glob.glob(voxel_dataset_path + '/*')
    voxels_name = os.listdir(voxel_dataset_path)
    voxels_hash = []
    for ele in voxels_name:
        h1 = ele.split('.')[0]
        voxels_hash.append(h1)

    voxels = np.zeros((len(voxels_path),) + (1, 32, 32, 32), dtype=np.float32)
    for i, name in enumerate(voxels_path):
        model = read_voxel_data(name)
        if padding:
            model = nd.zoom(model, (0.75, 0.75, 0.75), mode='constant', order=0)
            model = np.pad(model, ((4, 4), (4, 4), (4, 4)), 'constant')
        voxels[i] = model.astype(np.float32)
    return 3.0 * voxels - 1.0, voxels_hash