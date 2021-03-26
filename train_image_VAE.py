
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K

from image_VAE import get_image_VAE
from utils import data_IO, arg_parser, save_train, custom_loss, metrics
import sys, os, random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ConFig = tf.ConfigProto()
ConFig.gpu_options.allow_growth = True
session = tf.Session(config=ConFig)


def learning_rate_scheduler(epoch):
    # initial_learning_rate * decay_rate ^ (step / decay_steps)
    if epoch < 50:
        return 0.0002
    else:
        return 0.0002 * (0.96 ** ((epoch - 50) / 10))


def main(args):
    # Training on all categories
    category_list = ['04530566', '02933112', '03211117', '02691156', '04256520',
                     '04379243', '03691459', '04401088', '02828884', '02958343',
                     '03001627', '03636649', '04090263']
    processed_dataset_path = args.processed_dataset
    voxel_files_list, image_files_list, multicat_hash_id = data_IO.multicat_path_list(processed_dataset_path,
                                                                                      category_list,
                                                                                      use_mode='train')
    
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
    model = get_image_VAE(z_dim, use_pretrain = True)

    # Get model structures
    vol_inputs = model['vol_inputs']
    reconstructions = model['outputs']
    z_mean = model['img_z_mean']
    z_logvar = model['img_z_logvar']
    z = model['img_z']

    image_encoder = model['image_encoder']
    image_embedding_model = model['image_embedding_model']
    view_feature_aggregator = model['view_feature_aggregator']
    decoder = model['image_vae_decoder']
    image_vae = model['image_vae']

    # Loss functions
    loss_type = args.loss

    # kl-divergence
    kl_loss = custom_loss.kl_loss(z_mean, z_logvar)

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(custom_loss.weighted_binary_crossentropy(vol_inputs, K.clip(sigmoid(reconstructions), 1e-7, 1.0 - 1e-7)),
                      'float32')

    # Loss in betatc VAE
    tc_loss = (args.beta - 1.) * custom_loss.total_correlation(z, z_mean, z_logvar)

    # Add metrics
    # precision = metrics.get_precision(vol_inputs, reconstructions)
    IoU = metrics.get_IoU(vol_inputs, reconstructions)

    # opt = Adam(lr=learning_rate)
    opt = SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    ### Define loss function
    image_vae.add_loss(BCE_loss)
    image_vae.add_metric(BCE_loss, name='recon_loss', aggregation='mean')
    image_vae.add_metric(IoU, name='IoU', aggregation='mean')

    if loss_type == 'bce':
        print('Using VAE model without kl loss')
    elif loss_type == 'vae':
        print('Using VAE model')
        image_vae.add_loss(kl_loss)
        image_vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    elif loss_type == 'bvae':
        print('Using beta-VAE model')
        image_vae.add_loss(args.beta * kl_loss)
        image_vae.add_metric(args.beta * kl_loss, name='beta_kl_loss', aggregation='mean')
    elif loss_type == 'btcvae':
        print('Using beta-tc-VAE model')
        image_vae.add_loss(kl_loss)
        image_vae.add_loss(tc_loss)
        image_vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        image_vae.add_metric(tc_loss, name='tc_loss', aggregation='mean')
    image_vae.compile(optimizer= opt)

    plot_model(image_vae, to_file=os.path.join(model_pdf_path, 'image_vae.pdf'), show_shapes=True)
    plot_model(decoder, to_file=os.path.join(model_pdf_path, 'image_vae-decoder.pdf'), show_shapes=True)
    plot_model(image_encoder, to_file=os.path.join(model_pdf_path, 'image-encoder.pdf'), show_shapes=True)
    plot_model(image_embedding_model, to_file=os.path.join(model_pdf_path, 'image_embedding_model.pdf'),
               show_shapes=True)
    plot_model(view_feature_aggregator, to_file=os.path.join(model_pdf_path, 'feature_aggregator.pdf'),
               show_shapes=True)
    save_train.save_config_pro(save_path=train_data_path)

    def generate_batch_data(voxel_path_list,multicat_hash, batch_size):

        number_of_elements = len(voxel_path_list)
        random.shuffle(multicat_hash)

        voxel_file_path = []
        image_file_path = []

        for cat_hash in multicat_hash:
            category, hash = cat_hash.split('_')[0], cat_hash.split('_')[1]
            voxel_file = os.path.join(processed_dataset_path, category, 'voxel', 'train', hash)
            image_file = os.path.join(processed_dataset_path, category, 'image', 'train', hash)
            voxel_file_path.append(voxel_file)
            image_file_path.append(image_file)

        while 1:
            for start_idx in range(number_of_elements // batch_size):
                excerpt = slice(start_idx * batch_size, (start_idx + 1) * batch_size)

                image_one_batch_files = image_file_path[excerpt]
                image_one_batch = data_IO.imagePathList2matrix(image_one_batch_files)

                voxel_one_batch_files = voxel_file_path[excerpt]
                voxel_one_batch = data_IO.voxelPathList2matrix(voxel_one_batch_files)

                yield ([image_one_batch, voxel_one_batch],)

    train_callbacks = [
        #tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-7, cooldown=1),
        tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=0),
        tf.keras.callbacks.TensorBoard(log_dir=train_data_path),
        tf.keras.callbacks.CSVLogger(filename=train_data_path + '/training_log')
    ]

    image_vae.fit_generator(
        generate_batch_data(voxel_files_list,multicat_hash_id, batch_size),
        #steps_per_epoch=len(voxel_files_list) // batch_size,
        steps_per_epoch=5,
        epochs=epoch_num,
        callbacks=train_callbacks
    )

    image_embedding_model.save_weights(os.path.join(train_data_path, 'weightsEnd_viewFeatureEmbed.h5'))
    view_feature_aggregator.save_weights(os.path.join(train_data_path, 'weightsEnd_viewFeatureAggre.h5'))
    image_encoder.save_weights(os.path.join(train_data_path, 'weightsEnd_imgEncoder.h5'))
    decoder.save_weights(os.path.join(train_data_path, 'weightsEnd_voxDecoder.h5'))
    image_vae.save_weights(os.path.join(train_data_path, 'weightsEnd_all.h5'))

if __name__ == '__main__':
    main(arg_parser.parse_train_arguments(sys.argv[1:]))
