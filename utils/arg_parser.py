import argparse

def parse_train_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, choices=['deepmind_enc','vae','bvae', 'vae-0', 'btcvae'],
                        help='The encoder architecture to use', default='deepmind_enc')

    parser.add_argument('--data_dir', type=str,
                        help='Data directory.',
                        default=None)

    parser.add_argument('--graph_dir', type=str,
                        help='The directory to write the training graphs.',
                        default='../graphs/Celeb_A/')

    parser.add_argument('--save_dir', type=str,
                        help='The directory to save the trained model.',
                        default='../saved_models/')

    parser.add_argument('--test_image_folder', type=str,
                        help='The directory of the test images.',
                        default='../test_images/')

    parser.add_argument('--latent_vector_size', type=int,
                        help='The size of the embedding layers.',
                        default=200)

    parser.add_argument('--val_split', type=float,
                        help='The percentage of generated_data in the validation set',
                        default=0.2)

    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training.',
                        default=16)

    parser.add_argument('--val_batch_size', type=int,
                        help='Batch size for validation.',
                        default=64)

    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'SGD'],
                        help='The optimization algorithm to use', default='ADAM')

    parser.add_argument('--base_learning_rate', type=float,
                        help='The base learning rate for the model.',
                        default=5e-4)

    parser.add_argument('--num_epochs', type=int,
                        help='The total number of epochs for training.',
                        default=120)

    parser.add_argument('--scheduler_epoch', type=int,
                        help='The number of epochs to wait for the val loss to improve.',
                        default=10)

    parser.add_argument('--decay_factor', type=float,
                        help='The learning rate decay factor.',
                        default=0.1)

    parser.add_argument('--beta', type=float,
                        help='The vae regularizer.',
                        default=1.5)

    parser.add_argument('--capacity', type=float,
                        help='The latent space capacity.',
                        default=10.0)

    parser.add_argument('--max_epochs', type=float,
                        help='The maximum epoch to linearly increase the vae capacity.',
                        default=100)

    parser.add_argument('--num_workers', type=float,
                        help='The number of workers to use during training.',
                        default=8)

    parser.add_argument('--multi_process', type=bool,
                        help='Use multi-processing for dit generator during training.',
                        default=True)

    return parser.parse_args(argv)

def parse_test_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights_file', type=str,
                        help='the path of weights in .h5 file.',
                        default=None)

    parser.add_argument('--test_data_dir', type=str,
                        help='the path of test data in .binvox.',
                        default=None)

    parser.add_argument('--save_dir', type=str,
                        help='The directory to save the test data.',
                        default='../saved_models/')

    parser.add_argument('--save_ori', type=bool,
                        help='Save the original test data in the save_dir.',
                        default=True)

    parser.add_argument('--generate_img', type=bool,
                        help='Generate images from .binvox files',
                        default=True)

    return parser.parse_args(argv)