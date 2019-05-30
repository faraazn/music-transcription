import keras
from keras.layers import *
from keras import backend as K
from hparams import EMBED_SIZE, LR, F_BINS


def get_encoder():
    b_init = keras.initializers.Constant(value=0.1)
    # ENCODER
    spectrogram = Input(shape=(F_BINS, 64, 2))
    model = Conv2D(16, (1, 1), padding="same", bias_initializer=b_init)(spectrogram)
    model = Conv2D(16, (2, 4), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(32, (4, 2), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(1, 2))(model)
    
    model = Conv2D(32, (2, 4), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(32, (4, 2), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(1, 2))(model)
    
    model = Conv2D(64, (2, 4), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(64, (4, 2), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(4, 1))(model)
    
    model = Conv2D(128, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(128, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(4, 2))(model)
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(4, 2))(model)
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(4, 2))(model)
#     TODO: how to add minibatch std
    
    model = Conv2D(256, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(256, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    # ENCODED VECTOR
    model = Flatten()(model)
    model = Dense(EMBED_SIZE, bias_initializer=b_init)(model)
    model = Activation("sigmoid")(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    return keras.models.Model(spectrogram, model, name="encoder")


def get_dense_encoder():
    b_init = keras.initializers.Constant(value=0.1)
    # ENCODER
    spectrogram = Input(shape=(F_BINS, 64, 2))
    model = Flatten()(spectrogram)
    model = Dense(256, bias_initializer=b_init)(model)
    model = Activation("relu")(model)
    model = Dense(EMBED_SIZE, bias_initializer=b_init)(model)
    model = Activation("sigmoid")(model)
    
    return keras.models.Model(spectrogram, model, name="dense_encoder")
    
    
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

    
def get_vae_encoder():
    spectrogram = Input(shape=(F_BINS, 64, 1))
    model = get_small_1d_encoder()(spectrogram)
    
    z_mean = Dense(EMBED_SIZE)(model)
    z_mean = Activation("sigmoid")(z_mean)
    
    z_log_sigma = Dense(EMBED_SIZE)(model)
    z_log_sigma = Activation("sigmoid")(z_log_sigma)
    
    z = Lambda(sampling)([z_mean, z_log_sigma])
    
    return keras.models.Model(spectrogram, [z_mean, z_log_sigma, z], name="vae_encoder")
    

def get_decoder():
    b_init = keras.initializers.Constant(value=0.1)
    # DECODER
    latent = Input(shape=(EMBED_SIZE,))
    model = Reshape((1, 1, EMBED_SIZE))(latent)
    model = UpSampling2D(size=(4, 2))(model)
    model = Conv2D(64, (4, 2), padding="same", bias_initializer=b_init)(model)
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
    
    model = Conv2D(64, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(64, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(64, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(32, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(32, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 4))(model)
    
    model = Conv2D(16, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(1, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    spectrogram = Activation("tanh")(model)
    
    return keras.models.Model(latent, spectrogram, name="decoder")


def get_discriminator():
    spectrogram = Input(shape=(F_BINS, 128, 2))
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


def get_small_1d_encoder():
    b_init = keras.initializers.Constant(value=0.1)
    
    spectrogram = Input(shape=(F_BINS, 64, 1))
    model = Conv2D(16, (1, 3), padding="same", bias_initializer=b_init)(spectrogram)
    model = LeakyReLU(alpha=0.2)(model)
#     model = MaxPooling2D(pool_size=(1, 8))(model)
    model = Conv2D(16, (3, 3), padding="same", bias_initializer=b_init)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(32, 16))(model)
    
#     model = Conv2D(32, (3, 3), padding="same", bias_initializer=b_init)(spectrogram)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
#     model = Conv2D(32, (3, 1), padding="same", bias_initializer=b_init)(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = MaxPooling2D(pool_size=(8, 1))(model)
    
#     model = Conv2D(64, (3, 3), padding="same", bias_initializer=b_init)(spectrogram)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = Conv2D(64, (3, 3), padding="same", bias_initializer=b_init)(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = MaxPooling2D(pool_size=(16, 4))(model)
    
    model = Flatten()(model)
#     model = Dense(128, bias_initializer=b_init)(model)
#     model = LeakyReLU(alpha=0.2)(model)
    model = Dense(EMBED_SIZE, bias_initializer=b_init)(model)
    model = Activation("sigmoid")(model)
    
    return keras.models.Model(spectrogram, model, name="small_1d_encoder")


def get_classifier(encoder=None):
    if not encoder:
        encoder = get_small_1d_encoder()
        
    encoder.summary()
    
    adam = keras.optimizers.Adam(lr=LR)
    noisy_spectrogram = keras.layers.Input(shape=(F_BINS, 64, 1), name="spectrogram")
    latent_instr = encoder(noisy_spectrogram)
    
    onehot_instr = Dense(128)(latent_instr)
    onehot_instr = Activation("softmax")(onehot_instr)
    classifier = keras.models.Model(noisy_spectrogram, onehot_instr, name="classifier")
    classifier.compile(optimizer=adam, loss='categorical_crossentropy')
    
    return encoder, classifier


def get_autoencoder(encoder=None, decoder=None):
    if not encoder:
        assert not decoder
        encoder = get_small_1d_encoder()
        decoder = get_decoder()
    
    noisy_spectrogram = keras.layers.Input(shape=(F_BINS, 64, 1), name="spectrogram")
    latent_instr = encoder(noisy_spectrogram)
    reconstructed_instr = decoder(latent_instr)
    autoencoder = keras.models.Model(noisy_spectrogram, reconstructed_instr, name="autoencoder")
    
    adam = keras.optimizers.Adam(lr=LR)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')
    return encoder, decoder, autoencoder


def get_vae(encoder=None, decoder=None):
    if not encoder:
        assert not decoder
        encoder = get_vae_encoder()
        decoder = get_decoder()
    
    noisy_spectrogram = keras.layers.Input(shape=(F_BINS, 64, 1))
    z_mean, z_log_var, z = encoder(noisy_spectrogram)
    reconstructed_instr = decoder(z)
    
    def my_vae_loss(y_true, y_pred):
        xent_loss = keras.metrics.mse(K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss
    
    vae = keras.models.Model(noisy_spectrogram, reconstructed_instr)
    adam = keras.optimizers.Adam(lr=LR)
    vae.compile(optimizer=adam, loss=my_vae_loss)
    return encoder, decoder, vae, my_vae_loss
    

def get_gan():
    encoder = get_encoder()
    decoder = get_decoder()
    discriminator = get_discriminator()

    adam = keras.optimizers.Adam(lr=LR)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    n_disc_trainable = len(discriminator.trainable_weights)

    noisy_spectrogram = keras.layers.Input(shape=(F_BINS, 64, 2))

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