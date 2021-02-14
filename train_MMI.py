import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

from MMI import *
from utils import data_IO, arg_parser, save_train, custom_loss
import sys, os, glob

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
learning_rate_3 = 0.001
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
    z_dim = args.latent_vector_size

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
    sigma = model['sigma']
    z = model['z']

    encoder = model['MMI_encoder']
    decoder = model['MMI_decoder']
    MMI = model['MMI']

    # Loss functions
    loss_type = args.loss

    # kl-divergence
    kl_loss_term = kl_loss(mu, sigma)

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(K.mean(weighted_binary_crossentropy(vol_inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32')

    # Loss in betatc VAE
    z_edit = tf.expand_dims(z,0)
    tc_loss_term , tc = custom_loss.tc_term(args.beta, z_edit, mu, sigma)
    #tc_loss_term = tf.squeeze(tc_loss_term, axis=0)

    # Total loss
    if loss_type == 'bce':
        total_loss = BCE_loss
    elif loss_type == 'vae':
        print('Using VAE model')
        total_loss = BCE_loss + kl_loss_term
    elif loss_type == 'bvae':
        print('Using beta-VAE model')
        total_loss = BCE_loss + args.beta * kl_loss_term
    elif loss_type == 'btcvae':
        print('Using beta-tc-VAE model')
        total_loss = BCE_loss + kl_loss_term + tc_loss_term

    MMI.add_loss(total_loss)
    sgd = SGD(lr = learning_rate_1, momentum = momentum, nesterov = True)
    adam = Adam(lr=learning_rate_3)

    MMI.compile(optimizer = adam, metrics = ['accuracy'])

    plot_model(MMI, to_file = os.path.join(train_data_path,'MMI.pdf'), show_shapes = True)
    plot_model(encoder, to_file = os.path.join(train_data_path,'MMI-encoder.pdf'), show_shapes = True)
    plot_model(decoder, to_file = os.path.join(train_data_path,'MMI-decoder.pdf'), show_shapes = True)


    def generate_MMI_batch_data(voxel_path, image_path, batch_size):

        number_of_elements = len(os.listdir(voxel_path))
        hash_id = os.listdir(voxel_path)

        voxel_file_path = [os.path.join(voxel_path, id) for id in hash_id]
        image_file_path = [os.path.join(image_path, id) for id in hash_id]

        while 1:
            for start_idx in range(number_of_elements - batch_size):
                excerpt = slice(start_idx, start_idx + batch_size)

                image_one_batch_files = image_file_path[excerpt]
                images_one_batch = np.zeros((batch_size, 24) + g.IMAGE_SHAPE, dtype=np.float32)
                for i, element in enumerate(image_one_batch_files):
                    images_one_batch[i] = data_IO.imagepath2matrix(element)

                voxel_one_batch_files = voxel_file_path[excerpt]
                voxel_one_batch = np.zeros((batch_size,) + g.VOXEL_INPUT_SHAPE, dtype=np.float32)
                for i, element in enumerate(voxel_one_batch_files):
                    model = glob.glob(element + '/*')
                    model = data_IO.read_voxel_data(model[0])
                    voxel_one_batch[i] = model.astype(np.float32)
                voxel_one_batch = 3.0 * voxel_one_batch - 1.0

                yield [images_one_batch, voxel_one_batch], vol_inputs[excerpt]

    train_callbacks= [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2),
        tf.keras.callbacks.TensorBoard(log_dir=train_data_path),
        tf.keras.callbacks.CSVLogger(filename=train_data_path+'/training_log'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(train_data_path,'weights_{epoch:03d}_{loss:.4f}.h5'),
            save_weights_only=True,
            period=30
        )
    ]

    MMI.fit_generator(
        generate_MMI_batch_data(voxel_dataset_path,image_dataset_path, batch_size),
        steps_per_epoch= len(os.listdir(voxel_dataset_path)) // batch_size,
        #steps_per_epoch= 100,
        epochs = epoch_num,
        callbacks=train_callbacks
    )

    save_train.save_train_config(__file__, './run_training.sh',
                                 './MMI.py', './utils/arg_parser.py',
                                 './run_testing.sh',save_path= train_data_path)
    #MMI.save_weights(os.path.join(train_data_path,'weights.h5'))

if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
