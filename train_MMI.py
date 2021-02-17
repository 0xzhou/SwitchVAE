import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K

from MMI import *
from utils import data_IO, arg_parser, save_train, custom_loss
import sys, os

ConFig = tf.ConfigProto()
ConFig.gpu_options.allow_growth = True
session = tf.Session(config=ConFig)


def main(args):
    # Hyperparameters
    epoch_num = args.num_epochs
    batch_size = args.batch_size
    z_dim = args.latent_vector_size
    learning_rate = args.initial_learning_rate

    # Path configuration
    voxel_dataset_path = args.binvox_dir
    image_dataset_path = args.image_dir
    save_path = args.save_dir
    train_data_path = save_train.create_log_dir(save_path)

    # Model selection
    model = get_MMI(z_dim, 'switch')

    # Get model structures
    vol_inputs = model['vol_inputs']
    outputs = model['outputs']
    mu = model['mu']
    logvar = model['logvar']
    z = model['z']

    encoder = model['MMI_encoder']
    decoder = model['MMI_decoder']
    MMI = model['MMI']

    # Loss functions
    loss_type = args.loss

    # kl-divergence
    kl_loss_term = custom_loss.kl_loss(mu, logvar)

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(custom_loss.weighted_binary_crossentropy(vol_inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7)),
                      'float32')

    # Loss in betatc VAE
    z_edit = tf.expand_dims(z, 0)
    tc_loss_term, tc = custom_loss.tc_term(args.beta, z_edit, mu, 2 * logvar)
    # tc_loss_term = tf.squeeze(tc_loss_term, axis=0)

    adam = Adam(lr=learning_rate)
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    # Total loss
    if loss_type == 'bce':
        # total_loss = BCE_loss
        MMI.add_loss(BCE_loss)
        MMI.compile(optimizer=adam, metrics=['accuracy'])
    elif loss_type == 'vae':
        print('Using VAE model')
        # total_loss = BCE_loss + kl_loss_term
        MMI.add_loss(BCE_loss)
        MMI.add_loss(kl_loss_term)
        MMI.compile(optimizer=adam, metrics=['accuracy'])
        MMI.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        MMI.add_metric(kl_loss_term, name='kl_loss', aggregation='mean')
    elif loss_type == 'bvae':
        print('Using beta-VAE model')
        # total_loss = BCE_loss + args.beta * kl_loss_term
        MMI.add_loss(BCE_loss)
        MMI.add_loss(args.beta * kl_loss_term)
        MMI.compile(optimizer=adam, metrics=['accuracy'])
        MMI.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        MMI.add_metric(args.beta * kl_loss_term, name='beta_kl_loss', aggregation='mean')
    elif loss_type == 'btcvae':
        print('Using beta-tc-VAE model')
        # total_loss = BCE_loss + kl_loss_term + tc_loss_term
        MMI.add_loss(BCE_loss)
        MMI.add_loss(kl_loss_term)
        MMI.add_loss(tc_loss_term)
        MMI.compile(optimizer=adam, metrics=['accuracy'])
        MMI.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
        MMI.add_metric(kl_loss_term, name='kl_loss', aggregation='mean')
        MMI.add_metric(tc_loss_term, name='tc_loss', aggregation='mean')

    # MMI.add_loss(total_loss)
    # MMI.compile(optimizer=adam, metrics=['accuracy'])

    plot_model(MMI, to_file=os.path.join(train_data_path, 'MMI.pdf'), show_shapes=True)
    plot_model(encoder, to_file=os.path.join(train_data_path, 'MMI-encoder.pdf'), show_shapes=True)
    plot_model(decoder, to_file=os.path.join(train_data_path, 'MMI-decoder.pdf'), show_shapes=True)
    save_train.save_config_pro(save_path=train_data_path)

    def generate_MMI_batch_data(voxel_path, image_path, batch_size):

        number_of_elements = len(os.listdir(voxel_path))
        hash_id = os.listdir(voxel_path)

        voxel_file_path = [os.path.join(voxel_path, id) for id in hash_id]
        image_file_path = [os.path.join(image_path, id) for id in hash_id]

        while 1:
            for start_idx in range(number_of_elements - batch_size):
                excerpt = slice(start_idx, start_idx + batch_size)

                image_one_batch_files = image_file_path[excerpt]
                image_one_batch = data_IO.image_folder_list2matrix(image_one_batch_files)

                voxel_one_batch_files = voxel_file_path[excerpt]
                voxel_one_batch = data_IO.voxel_folder_list2matrix(voxel_one_batch_files)

                yield [image_one_batch, voxel_one_batch], vol_inputs[excerpt]

    train_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000001, cooldown=1),
        tf.keras.callbacks.TensorBoard(log_dir=train_data_path),
        tf.keras.callbacks.CSVLogger(filename=train_data_path + '/training_log'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(train_data_path, 'weights_{epoch:03d}_{loss:.4f}.h5'),
            save_weights_only=True,
            period=30
        )
    ]

    MMI.fit_generator(
        generate_MMI_batch_data(voxel_dataset_path, image_dataset_path, batch_size),
        steps_per_epoch=len(os.listdir(voxel_dataset_path)) // batch_size,
        # steps_per_epoch= 100,
        epochs=epoch_num,
        callbacks=train_callbacks
    )

    MMI.save_weights(os.path.join(train_data_path, 'end_weights_{epoch:03d}_{loss:.4f}.h5'))

if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
