import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

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
    if epoch < 50:
        return 0.0002
    else:
        return 0.0002 * (0.96 ** ((epoch - 50) / 10))

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
    if loss_type == 'bce':
        print('Using VAE model with only bce_loss')
        vae.add_loss(BCE_loss)
        vae.compile(optimizer=opt)
        vae.add_metric(IoU, name='IoU',aggregation='mean')
    elif loss_type == 'vae':
        print('Using VAE model')
        vae.add_loss(BCE_loss)
        vae.add_loss(kl_loss)
        vae.compile(optimizer=opt)
        vae.add_metric(IoU, name='IoU', aggregation='mean')
        vae.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    elif loss_type == 'bvae':
        print('Using beta-VAE model')
        vae.add_loss(BCE_loss)
        vae.add_loss(args.beta * kl_loss)
        vae.compile(optimizer=opt)
        vae.add_metric(IoU, name='IoU', aggregation='mean')
        vae.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        vae.add_metric(args.beta * kl_loss, name='beta_kl_loss', aggregation='mean')
    elif loss_type == 'btcvae':
        print('Using beta-tc-VAE model')
        vae.add_loss(BCE_loss)
        vae.add_loss(kl_loss)
        vae.add_loss(tc_loss)
        vae.compile(optimizer=opt)
        vae.add_metric(IoU, name='IoU', aggregation='mean')
        vae.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        vae.add_metric(tc_loss, name='tc_loss', aggregation='mean')

    plot_model(vae, to_file = 'vae.pdf', show_shapes = True)
    plot_model(encoder, to_file = os.path.join(train_data_path,'vae_encoder.pdf'), show_shapes = True)
    plot_model(decoder, to_file = os.path.join(train_data_path,'vae_decoder.pdf'), show_shapes = True)

    hash = os.listdir(voxel_dataset_path)
    voxel_folder_list = [os.path.join(voxel_dataset_path,id) for id in hash]
    data_train = data_IO.voxelPathList2matrix(voxel_folder_list)

    train_callbacks = [
        #tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000001, cooldown=1),
        tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=0),
        tf.keras.callbacks.TensorBoard(log_dir=train_data_path),
        tf.keras.callbacks.CSVLogger(filename=train_data_path + '/training_log'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(train_data_path, 'weights_{epoch:03d}_{loss:.4f}.h5'),
            save_weights_only=True,
            period=100
        )
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
