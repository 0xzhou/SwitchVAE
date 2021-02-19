import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

from VAE import *
from utils import data_IO, arg_parser, save_train, custom_loss
import sys, os

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
momentum = 0.9

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = learning_rate_2
    return lr

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
    mu = model['mu']
    print("The shape of mu", mu.shape)
    logvar = model['logvar']
    print("The shape of logvar", logvar.shape)
    z = model['z']
    print("The shape of z", z.shape)

    encoder = model['encoder']
    decoder = model['decoder']
    vae = model['vae']

    # kl-divergence
    kl_loss_term = custom_loss.kl_loss(mu, logvar)

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(K.mean(custom_loss.weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32')

    # Loss in betatc VAE
    #z_edit = tf.expand_dims(z,0)
    tc_loss_term , tc = custom_loss.tc_term(args.beta, z, mu, logvar)
    tc_loss_term = tf.squeeze(tc_loss_term, axis=0)

    adam = Adam(lr=learning_rate)

    # Total loss
    if loss_type == 'bce':
        print('Using VAE model with only bce_loss')
        vae.add_loss(BCE_loss)
        vae.compile(optimizer=adam, metrics=['accuracy'])
    elif loss_type == 'vae':
        print('Using VAE model')
        #total_loss = BCE_loss + kl_loss_term
        vae.add_loss(BCE_loss)
        vae.add_loss(kl_loss_term)
        vae.compile(optimizer=adam, metrics=['accuracy'])
        vae.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        vae.add_metric(kl_loss_term, name='kl_loss', aggregation='mean')
    elif loss_type == 'bvae':
        print('Using beta-VAE model')
        #total_loss = BCE_loss + args.beta * kl_loss_term
        vae.add_loss(BCE_loss)
        vae.add_loss(args.beta * kl_loss_term)
        vae.compile(optimizer=adam, metrics=['accuracy'])
        vae.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        vae.add_metric(args.beta * kl_loss_term, name='beta_kl_loss', aggregation='mean')
    elif loss_type == 'btcvae':
        print('Using beta-tc-VAE model')
        #total_loss = BCE_loss + kl_loss_term + tc_loss_term
        vae.add_loss(BCE_loss)
        vae.add_loss(kl_loss_term)
        vae.add_loss(tc_loss_term)
        vae.compile(optimizer = adam, metrics = ['accuracy'])
        vae.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        vae.add_metric(kl_loss_term, name='kl_loss', aggregation='mean')
        vae.add_metric(tc_loss_term, name='tc_loss', aggregation='mean')

    #vae.add_loss(total_loss)

    plot_model(vae, to_file = 'vae.pdf', show_shapes = True)
    plot_model(encoder, to_file = os.path.join(train_data_path,'vae_encoder.pdf'), show_shapes = True)
    plot_model(decoder, to_file = os.path.join(train_data_path,'vae_decoder.pdf'), show_shapes = True)

    hash = os.listdir(voxel_dataset_path)
    voxel_folder_list = [os.path.join(voxel_dataset_path,id) for id in hash]
    data_train = data_IO.voxelPathList2matrix(voxel_folder_list)

    train_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000001, cooldown=1),
        tf.keras.callbacks.TensorBoard(log_dir=train_data_path),
        tf.keras.callbacks.CSVLogger(filename=train_data_path + '/training_log'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(train_data_path, 'weights_{epoch:03d}_{loss:.4f}.h5'),
            save_weights_only=True,
            period=50
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
    vae.save_weights(os.path.join(train_data_path,'end_weights.h5'))

if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
