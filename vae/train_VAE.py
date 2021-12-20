import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

from VAE import *
from utils import data_IO, arg_parser, save_train, custom_loss, metrics
import sys, os

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
momentum = 0.9

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)


def learning_rate_scheduler(epoch):
    # initial_learning_rate * decay_rate ^ (step / decay_steps)
    init_lr = 1e-4
    if epoch < 100:
        return init_lr
    else:
        return init_lr * (0.96 ** ((epoch - 100) / 10))

alpha = K.variable(0.0)
class epoch_kl_weight_callback(Callback): # Cyclic Annealing Schedule
    def __init__(self, alpha):
        self.alpha = alpha
    def on_epoch_begin(self, epoch, logs = None):
        cycle = 200  # Epochs per annealing cycle
        if int(epoch) % cycle <=100:
            K.set_value(self.alpha, epoch/100.0)
        else:
            K.set_value(self.alpha, 1.0)

def main(args):

    # Hyperparameters
    epoch_num = args.num_epochs
    batch_size = args.batch_size
    z_dim = args.latent_vector_size
    learning_rate = args.initial_learning_rate

    # Path configuration
    loss_type = args.loss
    voxel_dataset_path = args.binvox_dir
    save_path = args.save_dir
    train_data_path = save_train.create_log_dir(save_path)

    # Model selection
    model = get_voxel_VAE(z_dim)
    
    # Get model structures
    inputs = model['inputs']
    outputs = model['outputs']
    z_mean = model['z_mean']
    z_logvar = model['z_logvar']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']
    vae = model['vae']

    # kl-divergence
    kl_loss = custom_loss.kl_loss(z_mean, z_logvar)

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(K.mean(custom_loss.weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32')

    # Loss in betatc VAE
    tc_loss = (args.beta - 1.) * custom_loss.total_correlation(z, z_mean, z_logvar)

    IoU = metrics.get_IoU(inputs, outputs)

    #opt = Adam(lr=learning_rate)
    opt = SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    # Total loss
    vae.add_loss(BCE_loss)
    vae.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
    vae.add_metric(IoU, name='IoU', aggregation='mean' )
    if loss_type == 'bce':
        print('Using VAE model without kl loss')
    elif loss_type == 'vae':
        print('Using VAE model')
        vae.add_loss(alpha * kl_loss)
        vae.add_metric(alpha * kl_loss, name='kl_loss', aggregation='mean')
    elif loss_type == 'bvae':
        print('Using beta-VAE model')
        vae.add_loss(args.beta * kl_loss)
        vae.add_metric(args.beta * kl_loss, name='beta_kl_loss', aggregation='mean')
    elif loss_type == 'btcvae':
        print('Using beta-tc-VAE model')
        vae.add_loss(kl_loss)
        vae.add_loss(tc_loss)
        vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        vae.add_metric(tc_loss, name='tc_loss', aggregation='mean')
    vae.compile(optimizer=opt)

    plot_model(vae, to_file = 'vae.pdf', show_shapes = True)
    plot_model(encoder, to_file = os.path.join(train_data_path,'vae_encoder.pdf'), show_shapes = True)
    plot_model(decoder, to_file = os.path.join(train_data_path,'vae_decoder.pdf'), show_shapes = True)

    hash = os.listdir(voxel_dataset_path)
    voxel_folder_list = [os.path.join(voxel_dataset_path,id) for id in hash]
    data_train = data_IO.voxelPathList2matrix(voxel_folder_list)

    train_callbacks = [
        #tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-7, cooldown=1),
        tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=0),
        tf.keras.callbacks.TensorBoard(log_dir=train_data_path),
        tf.keras.callbacks.CSVLogger(filename=train_data_path + '/training_log'),
        epoch_kl_weight_callback(alpha)
    ]

    vae.fit(
        data_train, outputs,
        epochs = epoch_num,
        batch_size = batch_size,
        validation_data = (data_train, None),
        callbacks=train_callbacks
    )

    save_train.save_config_pro(save_path=train_data_path)
    encoder.save_weights(os.path.join(train_data_path, 'weightsEnd_voxEncoder.h5'))
    decoder.save_weights(os.path.join(train_data_path, 'weightsEnd_voxDecoder.h5'))
    vae.save_weights(os.path.join(train_data_path, 'weightsEnd_all.h5'))

if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
