import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

from VAE import *
#from beta_VAE import *
from utils import npytar, binvox_IO, arg_parser, save_train, custom_loss
import glob, sys, os, shutil

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
momentum = 0.9

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

    model_name = args.model
    dataset = args.data_dir
    save_path = args.save_dir
    train_data_path = save_train.create_log_dir(save_path)


    model = get_model(z_dim= args.latent_vector_size)
    inputs = model['inputs']
    outputs = model['outputs']
    mu = model['mu']
    sigma = model['sigma']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']
    vae = model['vae']

    # Loss functions

    # kl-divergence
    kl_loss_term = kl_loss(mu, sigma)
    print('THe shape of kl_loss_term', kl_loss_term.shape)

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(K.mean(weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32')
    print('The shape of BCE loss is', BCE_loss.shape)

    z_edit = tf.expand_dims(z,0)
    tc_loss_term , tc = custom_loss.tc_term(args.beta, z_edit, mu, sigma)
    print('the shape of tc_loss_term', tc_loss_term.shape)
    #tc_loss_term = tf.squeeze(tc_loss_term, axis=0)

    # Total loss
    if model_name == 'vae-0':
        loss = BCE_loss
    if model_name == 'vae':
        print('Using VAE model')
        loss = BCE_loss + kl_loss_term
    elif model_name == 'bvae':
        print('Using beta-VAE model')
        loss = BCE_loss + args.beta * kl_loss_term
    elif model_name == 'btcvae':
        print('Using beta-tc-VAE model')
        loss = BCE_loss + kl_loss_term + tc_loss_term

    vae.add_loss(loss)
    sgd = SGD(lr = learning_rate_1, momentum = momentum, nesterov = True)
    vae.compile(optimizer = sgd, metrics = ['accuracy'])

    plot_model(vae, to_file = 'vae.pdf', show_shapes = True)

    data_train, hash = binvox_IO.voxelpath2matrix(dataset)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_data_path)

    vae.fit(
        data_train,
        epochs = epoch_num,
        batch_size = batch_size,
        validation_data = (data_train, None),
        callbacks=[tensorboard_callback])

    save_train.save_train_config(__file__, './run_training.sh','./VAE.py', './utils/arg_parser.py', save_path= train_data_path)
    plot_model(encoder, to_file = os.path.join(train_data_path,'vae_encoder.pdf'), show_shapes = True)
    plot_model(decoder, to_file = os.path.join(train_data_path,'vae_decoder.pdf'), show_shapes = True)
    vae.save_weights(os.path.join(train_data_path,'weights.h5'))

if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
