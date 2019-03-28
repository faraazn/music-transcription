import keras
import numpy as np
import pretty_midi
import librosa
import glob 
import os
import scipy

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


class NoteIsoSequence(keras.utils.Sequence):
    """Generate note iso data for training/validation."""
    def __init__(self, song_paths, sample_duration=44100, n_fft=1024, batch_size=64, shuffle=False, 
                 fs=44100, debug=False, sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2",
                 instr_indices=None, note_indices=None, mode='train'):
        self.song_paths = song_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fs = fs
        self.sf2_path = sf2_path
        self.sample_duration = sample_duration
        self.n_fft = n_fft
        self.debug = debug
        self.instr_indices = instr_indices
        self.note_indices = note_indices
        self.mode = mode
        
        self.pm = pretty_midi.PrettyMIDI(song_paths[0])
        self.pm_samples = self.pm.fluidsynth(fs=fs, sf2_path=sf2_path)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1 #int(np.floor(len(self.song_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
#         song_index = np.random.randint(len(self.song_paths))
#         song_path = self.song_paths[song_index]
#         pm = pretty_midi.PrettyMIDI(song_path)
        
#         wav_path = song_path[:-4] + ".wav"
#         pm_samples = scipy.io.wavfile.read(wav_path)
        
        if self.mode == 'train':
            num_instruments = len(self.pm.instruments)
            instr_indices = np.random.randint(num_instruments, size=self.batch_size)
            note_indices = []
            for instr_id in instr_indices:
                instrument = self.pm.instruments[instr_id]
                num_notes = len(instrument.notes)
                note_indices.append(np.random.randint(num_notes))
            X, y = self.process_batch(self.pm, self.pm_samples, instr_indices, note_indices)
        else:
            X, y = self.process_batch(self.pm, self.pm_samples, self.instr_indices, self.note_indices)
        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        pass

    def process_batch(self, pm, samples, instr_indices, note_indices):
        'Generates data containing batch_size samples' 
        num_instruments = len(pm.instruments)
        
        X = []
        y = []
        
        assert len(instr_indices) == len(note_indices)
        for i in range(len(instr_indices)):
            instr_id = instr_indices[i]
            instrument = pm.instruments[instr_id]
            note_id = note_indices[i]
            note = instrument.notes[note_id]
            sample_start = int(note.start * self.fs)

            padded_samples = samples[:]
            if len(padded_samples > sample_start+self.sample_duration):
                padded_samples = padded_samples[sample_start:sample_start+self.sample_duration]
            if len(padded_samples) < self.sample_duration:
                padded_samples = np.pad(padded_samples, (0, self.sample_duration-len(padded_samples)), 
                                        'constant', constant_values=(0,0))

            noisy_stft = librosa.core.stft(y=padded_samples, n_fft=self.n_fft)
            # convert complex numbers to magnitude and phase
            final_noisy = np.stack((np.abs(noisy_stft), np.angle(noisy_stft)), axis=2)
            annotation = np.zeros((1, final_noisy.shape[1], final_noisy.shape[2]))
            annotation[0,0,0] = note.pitch
            annotation[0,0,1] = note.end - note.start
            final_input = np.append(final_noisy, annotation, axis=0)
            
            
            pm_iso = pretty_midi.PrettyMIDI()
            iso_instrument = pretty_midi.Instrument(instrument.program, is_drum=instrument.is_drum)
            iso_note = pretty_midi.Note(note.velocity, note.pitch, 0.0, note.end-note.start)
            iso_instrument.notes = [iso_note]
            pm_iso.instruments = [iso_instrument]

            pm_iso_samples = pm_iso.fluidsynth(fs=self.fs, sf2_path=self.sf2_path)
            if len(pm_iso_samples) > self.sample_duration:
                pm_iso_samples = pm_iso_samples[:self.sample_duration]
            if len(pm_iso_samples) < self.sample_duration:
                pm_iso_samples = np.pad(pm_iso_samples, (0, self.sample_duration-len(pm_iso_samples)), 
                                        'constant', constant_values=(0,0))
            
            iso_stft = librosa.core.stft(y=pm_iso_samples, n_fft=self.n_fft)
            # convert complex numbers to magnitude and phase
            final_iso = np.stack((np.abs(iso_stft), np.angle(iso_stft)), axis=2)
            # sample_rate = 44100 sample_duration = 44100 n_fft = 1024 -> (513, 173, 2)
            # trim axis 1 and pad axis 2 to get 512 (2^9), 175 (5*5*7)
            final_iso = final_iso[:-1, :, :]  # shape (512, 173, 2)
            final_iso_pad = np.zeros((512, 2, 2))
            final_iso = np.concatenate((final_iso, final_iso_pad), axis=1)  # shape (512, 175, 2)

            X.append(final_input)
            y.append(final_iso)
            if self.debug:
                print(final_input.shape)
                print(final_iso.shape)
            
        X = np.array(X)
        y = np.array(y)

        return X, y

def get_model():
    # ENCODE
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(30, (3, 3), padding="same", input_shape=(514, 173, 2)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    model.add(keras.layers.MaxPooling2D(pool_size=(8, 4)))

    model.add(keras.layers.Conv2D(60, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    model.add(keras.layers.Conv2D(60, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    model.add(keras.layers.MaxPooling2D(pool_size=(5, 4)))

    model.add(keras.layers.Conv2D(100, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    model.add(keras.layers.Conv2D(100, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(400))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization())
    
    # ENCODED VECTOR
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Activation("sigmoid"))
    
    # DECODE
    model.add(keras.layers.Dense(400))
    model.add(keras.layers.Reshape((4, 1, 100)))
    model.add(keras.layers.UpSampling2D(size=(4, 5)))
    model.add(keras.layers.Conv2D(100, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    model.add(keras.layers.Conv2D(100, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    
    model.add(keras.layers.UpSampling2D(size=(4, 5)))
    model.add(keras.layers.Conv2D(60, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    model.add(keras.layers.Conv2D(60, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=2))
    
    model.add(keras.layers.UpSampling2D(size=(8, 7)))
    model.add(keras.layers.Conv2D(2, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    

    model.compile(optimizer="rmsprop", loss='categorical_crossentropy')
    
    return model