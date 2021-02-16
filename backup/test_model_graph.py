
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
