import keras
import numpy as np
import pretty_midi
import librosa
import glob 
import os
import scipy
from keras.layers import *
from keras import backend as K

program_map = { # map program to invalid id or collapse
  # pianos -> acoustic grand piano 
  1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
  # chromatic percussion -> xylophone
  9: 14, 10: 14, 11: 14, 12: 14, 13: 14, 14: 14, 15: 14, 16: 14,
  # organ, accordion, harmonica -> church organ
  17: 20, 18: 20, 19: 20, 20: 20, 21: 20, 22: 20, 23: 20, 24: 20,
  # guitar -> acoustic guitar (nylon)
  25: 25, 26: 25, 27: 25, 28: 25, 29: 25, 30: 25, 31: 25, 32: 25,
  # bass guitar -> acoustic bass guitar
  33: 33, 34: 33, 35: 33, 36: 33, 37: 33, 38: 33, 39: 33, 40: 33,
  # strings -> violin
  41: 41, 42: 41, 43: 41, 44: 41, 45: 41, 46: 41, 47: 41, 48: 41,
  # ensemble, choir, voice -> invalid
  49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0,
  # brass -> trumpet
  57: 57, 58: 57, 59: 57, 60: 57, 61: 57, 62: 57, 63: 57, 64: 57,
  # reeds -> alto saxophone
  65: 66, 66: 66, 67: 66, 68: 66, 69: 66, 70: 66, 71: 66, 72: 66,
  # pipe -> flute
  73: 74, 74: 74, 75: 74, 76: 74, 77: 74, 78: 74, 79: 74, 80: 74,
  # synth lead -> invalid
  81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0,
  # synth pad -> invalid
  89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0,
  # synth effects -> invalid
  97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0,
  # ethnic -> shamisen
  105: 107, 106: 107, 107: 107, 108: 107, 109: 107, 110: 107, 111: 107, 112: 107,
  # percussive -> invalid
  113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0,
  # sound effects -> invalid
  121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0
}
one_hot_map = {
    1: 1, 14: 2, 20: 3, 25: 4, 33: 5, 41: 6, 57: 7, 66: 8, 74: 9, 107: 10
}


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


class NoteIsoSequence(keras.utils.Sequence):
    """Generate note iso data for training/validation."""
    def __init__(self, song_paths, sample_duration=64000, n_fft=2048, batch_size=32, 
                 fs=32000, debug=False, sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2",
                 instr_indices=[], note_indices=[], song_indices=[], epsilon=0.00001,
                 notesounds_path="/home/faraaz/workspace/music-transcription/data/note_sounds/"):
        self.song_paths = song_paths  # wav files
        self.batch_size = batch_size
        self.fs = fs
        self.sf2_path = sf2_path
        self.sample_duration = sample_duration
        self.n_fft = n_fft
        self.debug = debug
        self.instr_indices = instr_indices
        self.note_indices = note_indices
        self.epsilon = epsilon
        self.notesounds_path = notesounds_path
        self.song_indices = song_indices
        self.instr_indices = instr_indices
        self.note_indices = note_indices
        
        self.notesounds = {}
        for filename in glob.iglob(self.notesounds_path + "*.wav"):
            program = filename.split(".")[0]
            program = program.split("/")[-1]
            _, notesound_samples = scipy.io.wavfile.read(filename)
            self.notesounds[int(program)] = notesound_samples
        
        if not song_indices or not instr_indices or not note_indices:
            num_songs = len(song_paths)
#             s_indices = np.random.choice(range(num_songs), size=num_songs, replace=False)
            s_indices = range(num_songs)
            for s_index in s_indices:
                song_path = song_paths[s_index]
                mid_path = song_path[:-4] + ".mid"
                pm = pretty_midi.PrettyMIDI(mid_path)
                num_instr = len(pm.instruments)
#                 print("num_instr {}".format(num_instr))
#                 i_indices = np.random.choice(range(num_instr), size=num_instr, replace=False)
                i_indices = range(num_instr)
                for i_index in i_indices:
                    instrument = pm.instruments[i_index]
#                     print("instrument {}".format(instrument))
                    assert instrument.is_drum == False
                    num_notes = len(instrument.notes)
#                     print("num_notes {}".format(num_notes))
#                     n_indices = np.random.choice(range(num_notes), size=num_notes, replace=False)
                    n_indices = range(num_notes)
#                     print("np.amax(n_indices) {}".format(np.amax(n_indices)))
                    for ii in range(len(n_indices)):
                        n_index = n_indices[ii]
                        self.song_indices.append(s_index)
                        self.instr_indices.append(i_index)
                        self.note_indices.append(n_index)
#                         if ii == 32:
#                             break
#                     print("len {}".format(len(self.note_indices)))
#                     break
                # to ensure 1 song per batch, round to nearest batch size
                data_size = len(self.note_indices)
                rounded_size = int(data_size / self.batch_size) * self.batch_size
                self.song_indices = self.song_indices[:rounded_size]
                self.instr_indices = self.instr_indices[:rounded_size]
                self.note_indices = self.note_indices[:rounded_size]
#                 print("self.song_indices {}".format(self.song_indices[-8:]))
#                 print("self.instr_indices {}".format(self.instr_indices[-8:]))
#                 print("self.note_indices {}".format(self.note_indices[-8:]))
#                 print("data_size {}".format(data_size))
#                 print("rounded_size {}".format(rounded_size))
        
        self.song_index = self.song_indices[0]
        song_path = self.song_paths[self.song_index]
        print(song_path)
        mid_path = song_path[:-4] + ".mid"
        self.pm = pretty_midi.PrettyMIDI(mid_path)
        _, self.pm_samples = scipy.io.wavfile.read(song_path)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.note_indices) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        song_indices = self.song_indices[self.batch_size*index:self.batch_size*(index+1)]
        instr_indices = self.instr_indices[self.batch_size*index:self.batch_size*(index+1)]
        note_indices = self.note_indices[self.batch_size*index:self.batch_size*(index+1)]
        song_index = song_indices[0]
#         print("index {}, song_index {}, {}".format(index, song_index, self.pm.get_end_time()))
        assert song_indices.count(song_index) == self.batch_size  # should all be the same
        
        if song_index != self.song_index:
            # changed song, need to reload wav and midi
            assert False
            self.song_index = song_index
            song_path = self.song_paths[self.song_index]
            mid_path = song_path[:-4] + ".mid"
            self.pm = pretty_midi.PrettyMIDI(mid_path)
            _, self.pm_samples = scipy.io.wavfile.read(song_path)
            
        
        X = []
        y = []
        
        for i in range(self.batch_size):
            instr_id = instr_indices[i]
            
#             instrument = self.pm.instruments[instr_id]
            try:
                instrument = self.pm.instruments[instr_id]
            except IndexError:
                print("instrumentt, instr id {}, num instr {}, instr indices {}, note_indices {}".format(
                      instr_id, len(self.pm.instruments), instr_indices, note_indices))
                print(self.pm.get_end_time())
                assert False
            note_id = note_indices[i]
            try:
                note = instrument.notes[note_id]
            except IndexError:
                print("note")
                print(note_id)
                print(instrument)
                print(len(instrument.notes))
                assert False
            sample_start = int(note.start * self.fs)

            padded_samples = self.pm_samples[:]
            if len(padded_samples > sample_start+self.sample_duration):
                padded_samples = padded_samples[sample_start:sample_start+self.sample_duration]
            if len(padded_samples) < self.sample_duration:
                padded_samples = np.pad(padded_samples, (0, self.sample_duration-len(padded_samples)), 
                                        'constant', constant_values=(0,0))

            noisy_stft = librosa.core.stft(y=padded_samples, n_fft=self.n_fft)  # shape [1025, 126]
            # convert complex numbers to magnitude and phase, then scale
            magnitude = np.abs(noisy_stft)
            phase = np.angle(noisy_stft) 
            log_magnitude = np.log(magnitude + self.epsilon)
            largest_magnitude = max(np.abs(np.amin(log_magnitude)), np.amax(log_magnitude))
            magnitude_scale_factor = largest_magnitude * 1.25  # will scale range to [-0.8, 0.8]
            assert magnitude_scale_factor > 0
            scaled_magnitude = log_magnitude / magnitude_scale_factor
            scaled_phase = phase / (np.pi * 1.25)  # will scale range to [-0.8, 0.8]
            final_noisy = np.stack((scaled_magnitude, scaled_phase), axis=2)  # shape [1025, 126, 2]
            final_noisy = final_noisy[:-1,:,:]  # shape [1024, 126, 2]
            
            annotation = np.zeros((final_noisy.shape[0], 2, final_noisy.shape[2]))  # shape [1024, 2, 2]
            annotation[note.pitch,0,0] = 1  # one hot encode pitch
            annotation[0,0,1] = min(self.sample_duration, note.end-note.start) / self.sample_duration  # range (0, 1]
            final_input = np.append(final_noisy, annotation, axis=1)  # shape [1024, 128, 2]
            
            
#             iso_file = self.notesounds_path + "{}.wav".format(instrument.program)
            pm_iso_samples = self.notesounds[instrument.program]  #scipy.io.wavfile.read(iso_file)
            if len(pm_iso_samples) > self.sample_duration:
                pm_iso_samples = pm_iso_samples[:self.sample_duration]
            if len(pm_iso_samples) < self.sample_duration:
                pm_iso_samples = np.pad(pm_iso_samples, (0, self.sample_duration-len(pm_iso_samples)), 
                                        'constant', constant_values=(0,0))
            
            iso_stft = librosa.core.stft(y=pm_iso_samples, n_fft=self.n_fft)  # shape [1025, 126]
            # convert complex numbers to magnitude and phase, then scale
            magnitude = np.abs(iso_stft)
            phase = np.angle(iso_stft)
            log_magnitude = np.log(magnitude + self.epsilon)
            largest_magnitude = max(np.abs(np.amin(log_magnitude)), np.amax(log_magnitude))
            magnitude_scale_factor = largest_magnitude * 1.25  # will scale range to [-0.8, 0.8]
            assert magnitude_scale_factor > 0
            scaled_magnitude = log_magnitude / magnitude_scale_factor
            scaled_phase = phase / (np.pi * 1.25)  # will scale range to [-0.8, 0.8]
            
            final_iso = np.stack((scaled_magnitude, scaled_phase), axis=2)  # shape [1025, 126, 2]
            final_iso = final_iso[:-1, :, :]  # shape (1024, 126, 2)
            final_iso_pad = np.zeros((final_iso.shape[0], 2, final_iso.shape[2]))  # TODO: 0 -> small number?
            final_iso = np.concatenate((final_iso, final_iso_pad), axis=1)  # shape (1024, 128, 2)

            assert not np.any(np.isnan(final_iso)) and not np.any(np.isnan(final_input))
            assert np.all(final_iso < 1) and np.all(final_iso > -1)
            assert np.all(final_input <= 1) and np.all(final_input > -1)
            
            X.append(final_input)
            y.append(final_iso)
            if self.debug:
                print(final_input.shape)
                print(final_iso.shape)
            
        X = np.array(X)
        y = np.array(y)

        return X, y
    
    
    def on_epoch_end(self):
        'Update indices after each epoch'
        pass


def get_encoder():
    # ENCODER
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
    
    # ENCODED VECTOR
    model = Flatten()(model)
    model = Dense(256)(model)
#     model = Activation("sigmoid")(model)
    
    return keras.models.Model(spectrogram, model, name="encoder")


def get_vae_encoder():
    # ENCODER
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
    
    # VARIATIONALLY ENCODED VECTOR
    model = Flatten()(model)
    
    z_mean = Dense(256)(model)
    z_mean = Activation("sigmoid")(z_mean)
    
    z_log_sigma = Dense(256)(model)
    z_log_sigma = Activation("sigmoid")(z_log_sigma)
    
    z = Lambda(sampling)([z_mean, z_log_sigma])
    
    return keras.models.Model(spectrogram, [z_mean, z_log_sigma, z])
    
    
def get_decoder():
    # DECODER
    latent = Input(shape=(256,))
    model = Reshape((1, 1, 256))(latent)
    model = UpSampling2D(size=(16, 2))(model)
    model = Conv2D(256, (16, 2), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    # TODO: how to do pixel normalization
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(256, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(128, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(128, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(64, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = UpSampling2D(size=(2, 2))(model)
    
    model = Conv2D(32, (3, 3), padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization(axis=2)(model)
    model = Conv2D(2, (3, 3), padding="same")(model)
#     model = LeakyReLU(alpha=0.2)(model)
#     model = Activation("relu")(model)
#     model = BatchNormalization(axis=2)(model)
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
    
    return keras.models.Model(spectrogram, prediction)


def get_autoencoder(encoder=None, decoder=None):
    if not encoder:
        assert not decoder
        encoder = get_encoder()
        decoder = get_decoder()
    
    noisy_spectrogram = keras.layers.Input(shape=(1024, 128, 2), name="spectrogram")
    latent_instr = encoder(noisy_spectrogram)
    reconstructed_instr = decoder(latent_instr)
    autoencoder = keras.models.Model(noisy_spectrogram, reconstructed_instr)
    
    adam = keras.optimizers.Adam(lr=0.001)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')
    return encoder, decoder, autoencoder

def get_vae():
    encoder = get_vae_encoder()
    decoder = get_decoder()
    noisy_spectrogram = keras.layers.Input(shape=(1024, 128, 2))
    z_mean, z_log_var, z = encoder(noisy_spectrogram)
    reconstructed_instr = decoder(z)
    
    def my_vae_loss(y_true, y_pred):
        xent_loss = keras.metrics.mse(K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss
    
    vae = keras.models.Model(noisy_spectrogram, reconstructed_instr)
    adam = keras.optimizers.Adam(lr=0.001)
    vae.compile(optimizer=adam, loss=my_vae_loss)
    return vae, my_vae_loss
    

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
