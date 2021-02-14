import tensorflow as tf

from utils import data_IO, arg_parser, save_train
import sys, os
from utils.model import *

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
momentum = 0.9

# ConFig=tf.ConfigProto()
# ConFig.gpu_options.allow_growth=True
# session=tf.Session(config=ConFig)
gpus= tf.config.list_physical_devices('GPU') # tf2.1版本该函数不再是experimental
print(gpus) # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
tf.config.experimental.set_memory_growth(gpus[0], True) # 其实gpus本身就只有一个元素


def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = learning_rate_2
    return lr

def main(args):

    # Hyperparameters
    epoch_num = args.num_epochs
    batch_size = args.batch_size
    beta = args.beta

    model_name = args.model
    dataset = args.data_dir
    save_path = args.save_dir
    train_data_path = save_train.create_log_dir(save_path)
    latent_dimension = args.latent_vector_size

    # Select model
    if model_name == 'vae-0':
        model = VAE(beta= 0, tc=False, latent_dims=latent_dimension)
    elif model_name == 'vae':
        print('------------Using VAE model--------------')
        model = VAE(beta= 1, tc=False, latent_dims=latent_dimension)
    elif model_name == 'bvae':
        print('------------Using beta-VAE model--------------')
        model = VAE(beta=beta, tc=False, latent_dims=latent_dimension)
    elif model_name == 'btcvae':
        print('------------Using beta-tc-VAE model--------------')
        model = VAE(beta=beta, tc=True, latent_dims=latent_dimension)
    else:
        raise NotImplementedError

    #sgd = SGD(lr=learning_rate_1, momentum=momentum, nesterov=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate_2), metrics=['accuracy'])

    # inputs = model['inputs']
    # outputs = model['outputs']
    # mu = model['mu']
    # sigma = model['sigma']
    # z = model['z']
    #
    # encoder = model['encoder']
    # decoder = model['decoder']
    # vae = model['vae']


    data_train, hash= data_IO.voxeldataset2matrix(dataset)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_data_path)

    model.fit(
        data_train,
        epochs = epoch_num,
        batch_size = batch_size,
        validation_data = (data_train, None),
        callbacks=[tensorboard_callback])

    save_train.save_train_config(__file__, './run_training.sh','./VAE.py', './utils/arg_parser.py', save_path= train_data_path)
    # plot_model(encoder, to_file = os.path.join(train_data_path,'vae_encoder.pdf'), show_shapes = True)
    # plot_model(decoder, to_file = os.path.join(train_data_path,'vae_decoder.pdf'), show_shapes = True)
    model.save_weights(os.path.join(train_data_path,'weights.h5'))

if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
