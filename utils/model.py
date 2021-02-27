import numpy as np

import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Conv2D, MaxPool2D, Dense, Dropout, Flatten, \
    Lambda, Reshape, Conv3DTranspose, AveragePooling2D, ZeroPadding2D, Activation, MaxPooling2D, Add, GRU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

from utils import globals as g


def sampling(args):
    z_mean, z_logvar = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    with open('./test_sampling.txt', 'w') as f:
        f.write(str(epsilon[0][0])+'\n')
        f.close()

    return z_mean + K.exp(z_logvar / 2.0) * epsilon


def get_voxel_encoder(z_dim=200):
    enc_in = Input(shape=g.VOXEL_INPUT_SHAPE, name='VoxEncoder_inputs')

    enc_conv1 = BatchNormalization(name='VoxEncoder_bn1')(Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                                                 padding='valid', kernel_initializer='glorot_normal',
                                                                 activation='elu',
                                                                 data_format='channels_first', name='VoxEncoder_conv1')(
        enc_in))

    enc_conv2 = BatchNormalization(name='VoxEncoder_bn2')(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                                                 padding='same', kernel_initializer='glorot_normal',
                                                                 activation='elu',
                                                                 data_format='channels_first', name='VoxEncoder_conv2')(
        enc_conv1))

    enc_conv3 = BatchNormalization(name='VoxEncoder_bn3')(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                                                 padding='valid', kernel_initializer='glorot_normal',
                                                                 activation='elu',
                                                                 data_format='channels_first', name='VoxEncoder_conv3')(
        enc_conv2))

    enc_conv4 = BatchNormalization(name='VoxEncoder_bn4')(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                                                 padding='same', kernel_initializer='glorot_normal',
                                                                 activation='elu',
                                                                 data_format='channels_first', name='VoxEncoder_conv4')(
        enc_conv3))

    enc_fc1 = BatchNormalization(name='VoxEncoder_bn_fc1')(Dense(units=343, kernel_initializer='glorot_normal',
                                                                 activation='elu', name='VoxEncoder_fc1')(
        Flatten(name='VoxEncoder_flatten1')(enc_conv4)))

    z_mean = BatchNormalization(name='VoxEncoder_bn_z_mean')(Dense(units=z_dim, kernel_initializer='glorot_normal',
                                                                   activation=None, name='VoxEncoder_z_mean')(enc_fc1))

    z_logvar = BatchNormalization(name='VoxEncoder_bn_z_logvar')(Dense(units=z_dim, kernel_initializer='glorot_normal',
                                                                       activation=None, name='VoxEncoder_z_logvar')(
        enc_fc1))

    z = Lambda(sampling, output_shape=(z_dim,), name='VoxEncoder_z')([z_mean, z_logvar])

    encoder = Model(enc_in, [z_mean, z_logvar, z], name='Voxel_Encoder')
    return encoder


def get_voxel_decoder(z_dim=200):
    dec_in = Input(shape=(z_dim,), name='VoxDecoder_inputs')

    dec_fc1 = BatchNormalization(name='VoxDecoder_bn_fc1')(Dense(units=343, kernel_initializer='glorot_normal',
                                                                 activation='elu', name='VoxDecoder_fc1')(dec_in))

    dec_unflatten = Reshape(target_shape=(1, 7, 7, 7), name='VoxDecoder_reshape1')(dec_fc1)

    dec_conv1 = BatchNormalization(name='VoxDecoder_bn1')(
        Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                        padding='same', kernel_initializer='glorot_normal',
                        activation='elu', name='VoxDecoder_conv1',
                        data_format='channels_first')(dec_unflatten))

    dec_conv2 = BatchNormalization(name='VoxDecoder_bn2')(
        Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                        padding='valid', kernel_initializer='glorot_normal',
                        activation='elu', name='VoxDecoder_conv2',
                        data_format='channels_first')(dec_conv1))

    dec_conv3 = BatchNormalization(name='VoxDecoder_bn3')(
        Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                        padding='same', kernel_initializer='glorot_normal',
                        activation='elu', name='VoxDecoder_conv3',
                        data_format='channels_first')(dec_conv2))

    dec_conv4 = BatchNormalization(name='VoxDecoder_bn4')(
        Conv3DTranspose(filters=8, kernel_size=(4, 4, 4), strides=(2, 2, 2),
                        padding='valid', kernel_initializer='glorot_normal',
                        activation='elu', name='VoxDecoder_conv4',
                        data_format='channels_first')(dec_conv3))

    dec_conv5 = BatchNormalization(beta_regularizer=l2(0.001), gamma_regularizer=l2(0.001), name='VoxDecoder_bn5') \
        (Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                         padding='same', kernel_initializer='glorot_normal',
                         data_format='channels_first', name='VoxDecoder_conv5', )(dec_conv4))

    decoder = Model(dec_in, dec_conv5, name='Voxel_Decoder')
    return decoder


def _split_inputs(inputs):
    """
    split inputs to NUM_VIEW input
    :param inputs: a Input with shape VIEWS_IMAGE_SHAPE
    :return: a list of inputs which shape is IMAGE_SHAPE
    """
    splited_views = []
    for i in range(0, g.NUM_VIEWS):
        splited_views.append(inputs[:, i, :, :, :])
    return splited_views


def _view_pool(views):
    """
    this is the ViewPooling in the paper
    :param views: the NUM_VIEWS outputs of CNN1
    """
    expanded = [K.expand_dims(view, 0) for view in views]
    concated = K.concatenate(expanded, 0)
    reduced = K.max(concated, 0)
    return reduced

def _view_features(views):
    view_features = [K.expand_dims(view, -2) for view in views]
    view_features = K.concatenate(view_features, -2)
    return  view_features


def _cnn_img(input_shape):
    """
    this is the CNN1 Network in paper
    :param input_shape: a image's shape, not the shape of a batch of image
    :return: a model object
    """
    inputs = keras.Input(shape=input_shape, name='SVCNN_inputs')

    # this two layers don't omit any parameter for showing how to define conv and pool layer
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(3, 3),
                   padding='valid', activation='relu', use_bias=True,
                   kernel_initializer='glorot_normal', name='SVCNN_conv1')(inputs)
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid',
                      name='SVCNN_pool1')(conv1)

    conv2 = Conv2D(256, (5, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_normal', name='SVCNN_conv2')(pool1)
    pool2 = MaxPool2D((2, 2), (2, 2), name='SVCNN_pool2')(conv2)

    conv3 = Conv2D(384, (3, 3), padding='same', activation='relu',
                   kernel_initializer='glorot_normal', name='SVCNN_conv3')(pool2)
    conv4 = Conv2D(384, (3, 3), padding='same', activation='relu',
                   kernel_initializer='glorot_normal', name='SVCNN_conv4')(conv3)

    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu',
                   kernel_initializer='glorot_normal', name='SVCNN_conv5')(conv4)
    pool5 = MaxPool2D((2, 2), (2, 2), name='SVCNN_pool5')(conv5)

    reshape = Flatten(name='SVCNN_flatten1')(pool5)

    cnn = keras.Model(inputs=inputs, outputs=reshape, name='SVCNN')
    return cnn

def _gru(feature_size):
    inputs = keras.Input(shape=(g.NUM_VIEWS, feature_size), name='View_Feature_Aggreator_input')
    output = GRU(units=feature_size, input_shape=(g.NUM_VIEWS, feature_size), name='GRU1')(inputs)
    gru =  keras.Model(inputs=inputs, outputs= output, name='View_Feature_Aggreator')
    return gru


def get_img_encoder(z_dim=200, use_resnet = True, use_gru = True):
    """
    input: Batch x Viewns x Width x Height x Channels (tensor)
    """
    # input placeholder with shape (None, 24, 137, 137, 3)
    inputs = Input(shape=g.VIEWS_IMAGE_SHAPE, name='MVCNN_input')

    # split inputs into views(a list), which has num_views elements, each element of views has shape (None, 137, 137, 3)
    views = Lambda(_split_inputs, name='MVCNN_split')(inputs)
    if use_resnet:
        cnn_model = get_resnet18()
    else:
        cnn_model = _cnn_img(g.IMAGE_SHAPE)

    view_pool = []

    if g.NUM_VIEWS == 1:
        pass
        # cnn_model_output = cnn_model(views)
        # cnn_model_output = AveragePooling2D((5,5))(cnn_model_output)
    else:
        for i, view in enumerate(views):
            single_view_features = cnn_model(view)
            view_pool.append(single_view_features)

        if use_gru:
            views_feature_aggregator = _gru(feature_size=1024)
            view_features = Lambda(_view_features, name='MVCNN_view_features')(view_pool)
            view_features = views_feature_aggregator(view_features)
        else:
            views_feature_aggregator = None
            view_features = Lambda(_view_pool(), name='MVCNN_view_features')(view_pool)

    # VAE sampling
    z_mean = BatchNormalization(name='MVCNN_bn_z_mean')(Dense(units=z_dim, kernel_initializer='glorot_normal', activation=None, name='MVCNN_z_mean')(view_features))
    z_logvar = BatchNormalization(name='MVCNN_bn_z_logvar')(Dense(units=z_dim, kernel_initializer='glorot_normal',activation=None, name='MVCNN_z_logvar')(view_features))
    z = Lambda(sampling, output_shape=(z_dim,), name='MVCNN_z')([z_mean, z_logvar])

    mvcnn_model = keras.Model(inputs=inputs, outputs=[z_mean, z_logvar, z], name='Image_MVCNN_VAE')

    return { 'mvcnn_model': mvcnn_model,
             'cnn_model': cnn_model,
             'view_feature_aggregator': views_feature_aggregator}


def get_resnet18():
    img_input = Input(shape=(137, 137, 3), name='data')
    x = BatchNormalization(name='bn_data',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': False})(img_input)

    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv0',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)
    x = BatchNormalization(name='bn0',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # (stage, rep) = [(0, 3), (1, 4), (2, 6), (3, 3)]
    # stage = 0, block = 0
    # x = ResidualBlock(64, 0, 0, strides=(1,1), cut='post', attention=Attention)(x)
    shortcut = Conv2D(64, (1, 1), name='stage1_unit1_sc', strides=(1, 1),
                      **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)
    x = BatchNormalization(name='stage1_unit1_bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage1_unit1_relu1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), name='stage1_unit1_conv1',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = BatchNormalization(name='stage1_unit1_bn2',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage1_unit1_relu2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, (3, 3), name='stage1_unit1_conv2',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = Add()([x, shortcut])

    # stage = 0, block = 1
    # x = ResidualBlock(64, stage, block, strides=(1, 1), cut='pre', attention=Attention)(x)
    shortcut = x
    x = BatchNormalization(name='stage1_unit2_bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage1_unit2_relu1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), name='stage1_unit2_conv1',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = BatchNormalization(name='stage1_unit2_bn2',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage1_unit2_relu2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, (3, 3), name='stage1_unit2_conv2',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = Add()([x, shortcut])

    # stage = 1, block = 0
    # x = ResidualBlock(128, stage, block, strides=(2, 2), cut='post', attention=Attention)(x)
    shortcut = Conv2D(128, (1, 1), name='stage2_unit1_sc', strides=(2, 2),
                      **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)
    x = BatchNormalization(name='stage2_unit1_bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage2_unit1_relu1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), name='stage2_unit1_conv1',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = BatchNormalization(name='stage2_unit1_bn2',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage2_unit1_relu2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, (3, 3), name='stage2_unit1_conv2',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = Add()([x, shortcut])
    # stage = 1, block = 1
    # x = ResidualBlock(128, stage, block, strides=(1, 1), cut='pre', attention=Attention)(x)
    shortcut = x
    x = BatchNormalization(name='stage2_unit2_bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage2_unit2_relu1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name='stage2_unit2_conv1',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = BatchNormalization(name='stage2_unit2_bn2',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage2_unit2_relu2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, (3, 3), name='stage2_unit2_conv2',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = Add()([x, shortcut])

    # stage = 2, block = 0
    # x = ResidualBlock(256, stage, block, strides=(2, 2), cut='post', attention=Attention)(x)
    shortcut = Conv2D(256, (1, 1), name='stage3_unit1_sc', strides=(2, 2),
                      **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)
    x = BatchNormalization(name='stage3_unit1_bn1',
                           **{'axis':-1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage3_unit1_relu1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), name='stage3_unit1_conv1',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = BatchNormalization(name='stage3_unit1_bn2',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage3_unit1_relu2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), name='stage3_unit1_conv2',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = Add()([x, shortcut])

    # stage = 2, block = 1
    # x = ResidualBlock(256, stage, block, strides=(1, 1), cut='pre', attention=Attention)(x)
    shortcut = x
    x = BatchNormalization(name='stage3_unit2_bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage3_unit2_relu1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), name='stage3_unit2_conv1',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = BatchNormalization(name='stage3_unit2_bn2',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage3_unit2_relu2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), name='stage3_unit2_conv2',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = Add()([x, shortcut])

    # stage = 3, block = 0
    # x = ResidualBlock(512, stage, block, strides=(2, 2), cut='post', attention=Attention)(x)
    shortcut = Conv2D(512, (1, 1), name='stage4_unit1_sc', strides=(2, 2),
                      **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)
    x = BatchNormalization(name='stage4_unit1_bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage4_unit1_relu1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(2, 2), name='stage4_unit1_conv1',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = BatchNormalization(name='stage4_unit1_bn2',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage4_unit1_relu2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(512, (3, 3), name='stage4_unit1_conv2',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = Add()([x, shortcut])

    # stage = 3, block = 1
    # x = ResidualBlock(512, stage, block, strides=(1, 1), cut='pre', attention=Attention)(x)
    shortcut = x
    x = BatchNormalization(name='stage4_unit2_bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage4_unit2_relu1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), name='stage4_unit2_conv1',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = BatchNormalization(name='stage4_unit2_bn2',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(
        x)
    x = Activation('relu', name='stage4_unit2_relu2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(512, (3, 3), name='stage4_unit2_conv2',
               **{'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'})(x)

    x = Add()([x, shortcut])

    x = BatchNormalization(name='bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(x)
    x = Activation('relu', name='relu1')(x)

    #x = AveragePooling2D((5,5), name='GRU_AP1')(x)
    x = Flatten(name='resnet_flatten1')(x)
    x = BatchNormalization(name='resnet_bn1',
                           **{'axis': -1, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True})(Dense(units=4096, name='resnet_fc1')(x))
    x = Dense(units=1024, name='resnet_fc2')(x)


    resnet18 = Model(inputs=img_input, outputs=x, name='ResNet18')
    #resnet18.load_weights('./utils/resnet18_imagenet_1000_no_top.h5', by_name=True)
    #plot_model(resnet18, to_file='./resnet18.pdf', show_shapes=True)

    #print(resnet18.summary())

    return  resnet18
