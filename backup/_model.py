def get_voxel_encoder_old(z_dim = 200):
    enc_in = Input(shape=g.VOXEL_INPUT_SHAPE)

    enc_conv1 = BatchNormalization()(
        Conv3D(
            filters=8,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='valid',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(enc_in))
    enc_conv2 = BatchNormalization()(
        Conv3D(
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(enc_conv1))
    enc_conv3 = BatchNormalization()(
        Conv3D(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='valid',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(enc_conv2))
    enc_conv4 = BatchNormalization()(
        Conv3D(
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(enc_conv3))

    enc_fc1 = BatchNormalization()(
        Dense(
            units=343,
            kernel_initializer='glorot_normal',
            activation='elu')(Flatten()(enc_conv4)))
    mu = BatchNormalization()(
        Dense(
            units=z_dim,
            kernel_initializer='glorot_normal',
            activation=None)(enc_fc1))
    sigma = BatchNormalization()(
        Dense(
            units=z_dim,
            kernel_initializer='glorot_normal',
            activation=None)(enc_fc1))
    z = Lambda(
        sampling,
        output_shape=(z_dim,))([mu, sigma])

    #encoder = Model(enc_in, [mu, sigma, z], name='Voxel_Variational_Encoder')
    encoder = Model(enc_in, [mu, sigma, z], name='Voxel_VAE')
    return encoder

def get_voxel_decoder_old(z_dim = 200):
    dec_in = Input(shape=(z_dim,))

    dec_fc1 = BatchNormalization()(
        Dense(
            units=343,
            kernel_initializer='glorot_normal',
            activation='elu')(dec_in))
    dec_unflatten = Reshape(
        target_shape=(1, 7, 7, 7))(dec_fc1)

    dec_conv1 = BatchNormalization()(
        Conv3DTranspose(
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(dec_unflatten))
    dec_conv2 = BatchNormalization()(
        Conv3DTranspose(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='valid',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(dec_conv1))
    dec_conv3 = BatchNormalization()(
        Conv3DTranspose(
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(dec_conv2))
    dec_conv4 = BatchNormalization()(
        Conv3DTranspose(
            filters=8,
            kernel_size=(4, 4, 4),
            strides=(2, 2, 2),
            padding='valid',
            kernel_initializer='glorot_normal',
            activation='elu',
            data_format='channels_first')(dec_conv3))
    dec_conv5 = BatchNormalization(
        beta_regularizer=l2(0.001),
        gamma_regularizer=l2(0.001))(
        Conv3DTranspose(
            filters=1,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            kernel_initializer='glorot_normal',
            data_format='channels_first')(dec_conv4))

    decoder = Model(dec_in, dec_conv5, name= 'Voxel_Generator')
    return decoder