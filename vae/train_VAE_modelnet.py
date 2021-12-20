from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

from utils import data_IO, arg_parser, save_train, custom_loss, metrics
from VAE import *
import sys, os, random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ConFig = tf.ConfigProto()
ConFig.gpu_options.allow_growth = True
session = tf.Session(config=ConFig)


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
    # Training on all categories
    modelnet_voxel_dataset = args.modelnet_voxel_dataset


    ModelNet10_CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                          'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    ModelNet40_CLASSES = [ 'airplane', 'bowl', 'table', 'chair', 'vase', 'glass_box', 'bathtub', 'toilet', 'range_hood',
                          'flower_pot', 'laptop', 'plant', 'cup', 'person', 'tent', 'sofa', 'monitor', 'keyboard',
                          'desk', 'mantel', 'curtain', 'bed', 'lamp', 'bench', 'dresser','car', 'sink',
                          'night_stand', 'stool', 'door', 'guitar', 'stairs', 'radio', 'tv_stand', 'cone', 'xbox',
                          'wardrobe', 'bookshelf', 'bottle', 'piano']


    multi_category_id = data_IO.generate_modelnet_idList(modelnet_voxel_dataset,ModelNet10_CLASSES,'train')

    # Hyperparameters
    epoch_num = args.num_epochs
    batch_size = args.batch_size
    z_dim = args.latent_vector_size
    learning_rate = args.initial_learning_rate

    # Path configuration
    save_path = args.save_dir
    train_data_path = save_train.create_log_dir(save_path)
    model_pdf_path = os.path.join(train_data_path, 'model_pdf_train')
    os.makedirs(model_pdf_path)

    # Model selection
    model = get_voxel_VAE(z_dim)

    # Get model structures
    vol_inputs = model['inputs']
    outputs = model['outputs']
    z = model['z']
    z_mean = model['z_mean']
    z_logvar = model['z_logvar']

    encoder = model['encoder']
    decoder = model['decoder']
    vae = model['vae']


    # Loss functions
    loss_type = args.loss

    # kl-divergence
    kl_loss = custom_loss.kl_loss(z_mean, z_logvar)

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(custom_loss.weighted_binary_crossentropy(vol_inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7)),
                      'float32')

    # Loss in betatc VAE
    tc_loss = (args.beta - 1.) * custom_loss.total_correlation(z, z_mean, z_logvar)

    # Add metrics
    # precision = metrics.get_precision(vol_inputs, outputs)
    IoU = metrics.get_IoU(vol_inputs, outputs)

    # opt = Adam(lr=learning_rate)
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

    plot_model(vae, to_file=os.path.join(model_pdf_path, 'model.pdf'), show_shapes=True)
    plot_model(encoder, to_file=os.path.join(model_pdf_path, 'encoder.pdf'), show_shapes=True)
    plot_model(decoder, to_file=os.path.join(model_pdf_path, 'decoder.pdf'), show_shapes=True)
    save_train.save_config_pro(save_path=train_data_path)

    def generate_batch_data(voxel_dataset,multicat_id, batch_size):

        number_of_elements = len(multicat_id)
        random.shuffle(multicat_id)

        while 1:
            for start_idx in range(number_of_elements // batch_size):
                excerpt = slice(start_idx * batch_size, (start_idx + 1) * batch_size)

                category_id_one_batch = multicat_id[excerpt]

                voxel_one_batch = np.zeros((batch_size,) + g.VOXEL_INPUT_SHAPE, dtype=np.float32)

                for i, cat_id in enumerate(category_id_one_batch):
                    category, hash = cat_id.rsplit('_',1)[0], cat_id.rsplit('_',1)[1]

                    voxel_one_batch_file = os.path.join(voxel_dataset,category,'train', cat_id+'.binvox')
                    voxel_one_batch[i] = data_IO.read_voxel_data(voxel_one_batch_file)

                yield (voxel_one_batch, )

    train_callbacks = [
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-7, cooldown=1),
        tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=0),
        tf.keras.callbacks.TensorBoard(log_dir=train_data_path),
        tf.keras.callbacks.CSVLogger(filename=train_data_path + '/training_log'),
        epoch_kl_weight_callback(alpha)
    ]

    vae.fit_generator(
        generate_batch_data(modelnet_voxel_dataset,multi_category_id, batch_size),
        steps_per_epoch=len(multi_category_id) // batch_size,
        #steps_per_epoch=5,
        epochs=epoch_num,
        callbacks=train_callbacks
    )
    encoder.save_weights(os.path.join(train_data_path, 'weightsEnd_voxEncoder.h5'))
    decoder.save_weights(os.path.join(train_data_path, 'weightsEnd_voxDecoder.h5'))
    vae.save_weights(os.path.join(train_data_path, 'weightsEnd_all.h5'))


if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
