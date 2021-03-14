import numpy as np
import shutil, sys, os, pickle
sys.path.append("..")

from MMI import *
from VAE import *
from utils import save_volume, data_IO, arg_parser, model
from utils import model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)


def main():

    latent_dims = 128

    interpolation_save_path = '/home/zmy/Downloads/bothTrain_lessFC_uniLoss1/voxel_latent_dict/chair_arith60'
    latent_vector_file1 = open('/home/zmy/Downloads/bothTrain_lessFC_uniLoss1/voxel_latent_dict/voxel_latent_dict_chair_all.pkl', 'rb')
    latent_vector_file2 = open('/home/zmy/Downloads/allCategory_uniLoss1/voxel_latent_dict/latent_dict_chair_train.pkl', 'rb')
    weights_path = '/home/zmy/Downloads/bothTrain_lessFC_uniLoss1/weightsEnd_voxDecoder.h5'

    object1 = '928678562a6a57db698a862d11861eaa'
    object2 = '4e3cc3f6d2c25b904addf6a5f8726274'
    object3 = 'c47f71319ead4eb8a4fb72f4f3b0e317'

    # Get the latent vector of two objects
    latent_space = np.zeros(shape=(4,128), dtype=np.float32)
    latent_vector_dict1 = pickle.load(latent_vector_file1)
    latent_vector_dict2 = pickle.load(latent_vector_file2)

    p1, p2, p3 = latent_vector_dict1[object1+'_z_mean'], latent_vector_dict1[object2+'_z_mean'],latent_vector_dict1[object3+'_z_mean']
    p4 = p1 - p2 + p3
    latent_space[0] = p1
    latent_space[1] = p2
    latent_space[2] = p3
    latent_space[3] = p4

    # Define the decoder model
    decoder = model.get_voxel_decoder(latent_dims)
    decoder.load_weights(weights_path,by_name=True)
    reconstructions = decoder.predict(latent_space)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(interpolation_save_path):
        os.makedirs(interpolation_save_path)

    shutil.copy2(__file__, interpolation_save_path)
    for i in range(reconstructions.shape[0]):
        name = str(i)
        save_volume.save_binvox_output(reconstructions[i, 0, :], name, interpolation_save_path, '_gen', save_bin=False,
                                       save_img=True)
if __name__ == '__main__':
    main()