import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

from VAE import *
from MMI import *
#from beta_VAE import *
from utils import npytar, binvox_IO, arg_parser, save_train, custom_loss, dataset_pre
import glob, sys, os, shutil

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
momentum = 0.9
image_shape=(137,137,3)


ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)


def weighted_binary_crossentropy(target, output):
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss

def kl_loss(z_mean,z_log_sigma_square):
    return - 0.5 * K.mean(1 + z_mean - K.square(z_mean) - K.exp(z_log_sigma_square))

def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = learning_rate_2
    return lr

def main(args):

    # Hyperparameters
    epoch_num = args.num_epochs
    batch_size = args.batch_size
    z_dim = args.latent_vector_size

    # Path configuration
    voxel_dataset_path = args.binvox_dir
    image_dataset_path = args.image_dir
    save_path = args.save_dir
    train_data_path = save_train.create_log_dir(save_path)

    # Model selection
    model = get_MMI(200, 'switch')

    # Get model structures
    vol_inputs = model['vol_inputs']
    outputs = model['outputs']
    #mu = model['mu']
    #sigma = model['sigma']
    z = model['z']

    encoder = model['MMI_encoder']
    decoder = model['MMI_decoder']
    MMI = model['MMI']

    # Loss functions
    loss_type = args.loss

    # kl-divergence
    #kl_loss_term = kl_loss(mu, sigma)

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(K.mean(weighted_binary_crossentropy(vol_inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32')

    # Loss in betatc VAE
    #z_edit = tf.expand_dims(z,0)
    #tc_loss_term , tc = custom_loss.tc_term(args.beta, z_edit, mu, sigma)
    #tc_loss_term = tf.squeeze(tc_loss_term, axis=0)

    # Total loss
    if loss_type == 'bce':
        total_loss = BCE_loss
    # elif model_name == 'vae':
    #     print('Using VAE model')
    #     total_loss = BCE_loss + kl_loss_term
    # elif model_name == 'bvae':
    #     print('Using beta-VAE model')
    #     total_loss = BCE_loss + args.beta * kl_loss_term
    # elif model_name == 'btcvae':
    #     print('Using beta-tc-VAE model')
    #     total_loss = BCE_loss + kl_loss_term + tc_loss_term

    MMI.add_loss(total_loss)
    sgd = SGD(lr = learning_rate_1, momentum = momentum, nesterov = True)
    MMI.compile(optimizer = sgd, metrics = ['accuracy'])

    plot_model(MMI, to_file = os.path.join(train_data_path,'MMI.pdf'), show_shapes = True)
    plot_model(encoder, to_file = os.path.join(train_data_path,'MMI-encoder.pdf'), show_shapes = True)
    plot_model(decoder, to_file = os.path.join(train_data_path,'MMI-decoder.pdf'), show_shapes = True)

    # Load train data
    voxel_data_train, hash = binvox_IO.voxelpath2matrix(voxel_dataset_path) # Number of element * 1 * 32 * 32 * 32
    image_path_list = [os.path.join(image_dataset_path,id) for id in hash] # Get the path list which corresponds with the objects in voxel_data_train

    def generate_batch_data(voxel_input =None, image_path_list=None, batch_size =None):
        while 1:
            for start_idx in range(len(voxel_input)- batch_size):
                excerpt = slice(start_idx, start_idx + batch_size)

                image_path_onebatch = image_path_list[excerpt]
                batch_images = np.zeros((batch_size,24) + image_shape, dtype=np.float32)
                for i, object in enumerate(image_path_onebatch):
                    batch_images[i] = dataset_pre.imagepath2matrix(object)

                yield [batch_images, voxel_input[excerpt]], vol_inputs[excerpt]

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_data_path)

    MMI.fit_generator(
        generate_batch_data(voxel_data_train, image_path_list, batch_size),
        steps_per_epoch= len(voxel_data_train) // batch_size,
        epochs = epoch_num,
        callbacks=[LearningRateScheduler(learning_rate_scheduler),tensorboard_callback])

    save_train.save_train_config(__file__, './run_training.sh', './VAE.py', './utils/arg_parser.py','./run_testing.sh',save_path= train_data_path)
    MMI.save_weights(os.path.join(train_data_path,'weights.h5'))

if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
