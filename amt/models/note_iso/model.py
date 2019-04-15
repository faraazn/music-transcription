import keras
from keras.layers import *
from keras import backend as K

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_encoder():
    # ENCODER
    spectrogram = Input(shape=(1024, 64, 2))
    model = Conv2D(16, (1, 1), padding="same")(spectrogram)
    model = Conv2D(16, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(4, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(4, 4))(model)
    
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
    
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
    # TODO: how to add minibatch std
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    # ENCODED VECTOR
    model = Flatten()(model)
    model = Dense(64)(model)
    model = Activation("sigmoid")(model)
    
    return keras.models.Model(spectrogram, model, name="encoder")


def get_vae_encoder():
    # ENCODER
    spectrogram = Input(shape=(1024, 64, 2))
    model = Conv2D(16, (1, 1), padding="same")(spectrogram)
    model = Conv2D(16, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(16, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(4, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(4, 4))(model)
    
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
    
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
    # TODO: how to add minibatch std
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    # VARIATIONALLY ENCODED VECTOR
    model = Flatten()(model)
    
    z_mean = Dense(64)(model)
    z_mean = Activation("sigmoid")(z_mean)
    
    z_log_sigma = Dense(64)(model)
    z_log_sigma = Activation("sigmoid")(z_log_sigma)
    
    z = Lambda(sampling)([z_mean, z_log_sigma])
    
    return keras.models.Model(spectrogram, [z_mean, z_log_sigma, z], name="vae_encoder")
    
    
def get_decoder():
    # DECODER
    latent = Input(shape=(64,))
    model = Reshape((1, 1, 64))(latent)
    model = UpSampling2D(size=(16, 2))(model)
    model = Conv2D(64, (16, 2), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    # TODO: how to do pixel normalization
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = BatchNormalization(axis=2)(model)
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = BatchNormalization(axis=2)(model)
#     model = UpSampling2D(size=(2, 2))(model)
    
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = BatchNormalization(axis=2)(model)
#     model = Conv2D(256, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = BatchNormalization(axis=2)(model)
#     model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(4, 2))(model)
    
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(4, 4))(model)
    
    model = Conv2D(16, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(2, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    spectrogram = Activation("tanh")(model)
    
    return keras.models.Model(latent, spectrogram, name="decoder")


def get_discriminator():
    spectrogram = Input(shape=(1024, 128, 2))
    model = Conv2D(32, (1, 1), padding="same")(spectrogram)
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(128, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(128, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    # TODO: how to add minibatch std
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    # predict real or fake
    model = Flatten()(model)
    model = Dense(1)(model)
    prediction = Activation("sigmoid")(model)
    
    return keras.models.Model(spectrogram, prediction, name="discriminator")


def get_autoencoder(encoder=None, decoder=None):
    if not encoder:
        assert not decoder
        encoder = get_encoder()
        decoder = get_decoder()
    
    noisy_spectrogram = keras.layers.Input(shape=(1024, 64, 2), name="spectrogram")
    latent_instr = encoder(noisy_spectrogram)
    reconstructed_instr = decoder(latent_instr)
    autoencoder = keras.models.Model(noisy_spectrogram, reconstructed_instr)
    
    adam = keras.optimizers.Adam(lr=0.001)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')
    return encoder, decoder, autoencoder

def get_vae(encoder, decoder):
    if not encoder:
        assert not decoder
        encoder = get_vae_encoder()
        decoder = get_decoder()
    
    noisy_spectrogram = keras.layers.Input(shape=(1024, 64, 2))
    z_mean, z_log_var, z = encoder(noisy_spectrogram)
    reconstructed_instr = decoder(z)
    
    def my_vae_loss(y_true, y_pred):
        xent_loss = keras.metrics.mse(K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss
    
    vae = keras.models.Model(noisy_spectrogram, reconstructed_instr)
    adam = keras.optimizers.Adam(lr=0.001, clipnorm=1.)
    vae.compile(optimizer=adam, loss=my_vae_loss)
    return encoder, decoder, vae, my_vae_loss
    

def get_gan():
    encoder = get_encoder()
    decoder = get_decoder()
    discriminator = get_discriminator()

    adam = keras.optimizers.Adam(lr=0.01)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    n_disc_trainable = len(discriminator.trainable_weights)

    noisy_spectrogram = keras.layers.Input(shape=(1024, 128, 2))

    latent_instr = encoder(noisy_spectrogram)
    reconstructed_instr = decoder(latent_instr)

    discriminator.trainable = False
    predicted_instr = discriminator(reconstructed_instr)

    autoencoder = keras.models.Model(noisy_spectrogram, reconstructed_instr)
    n_gen_trainable = len(autoencoder.trainable_weights)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')

    combined = keras.models.Model(noisy_spectrogram, predicted_instr)
    combined.compile(loss='binary_crossentropy', optimizer=adam)
    
    return autoencoder, discriminator, combined