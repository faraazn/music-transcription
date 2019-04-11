import keras
import numpy as np
import glob 
import os
from train_util import NoteIsoSequence
from train_util import get_autoencoder
import time

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
wav_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.wav'))
wav_files = [wav_file for wav_file in wav_files]
num_wav_files = len(wav_files)
train_wav_files = wav_files[:num_wav_files//5]
test_wav_files = wav_files[num_wav_files//5:]
wav_file = wav_files[0]

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
sample_duration = 2 * sample_rate
n_fft = 2048
batch_size = 4
epsilon = 0.00001

epochs = 1
workers = 1
use_multiprocessing = False
validation_steps = 10
steps_per_epoch = None

song_indices = [] #[0, 0, 0]*32*10 #[0]*64*10
instr_indices = [] #[0, 1, 2]*32*10 #[0]*64*10
note_indices = [] #[0, 50, 100]*32*10 #[0]*64*10

encoder = None
decoder = None

if __name__ == "__main__":
    print(wav_file)
    train_generator = NoteIsoSequence(train_wav_files, sample_duration=sample_duration, 
                                      fs=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)
#     valid_generator = NoteIsoSequence(test_wav_files, sample_duration=sample_duration, 
#                                       fs=sample_rate, n_fft=n_fft, batch_size=batch_size,
#                                       epsilon=epsilon, song_indices=song_indices, 
#                                       instr_indices=instr_indices, note_indices=note_indices)
    
    encoder = keras.models.load_model("ae/encoder-1.h")
    decoder = keras.models.load_model("ae/decoder-1.h")
    encoder, decoder, autoencoder = get_autoencoder(encoder, decoder)
    autoencoder.summary()

    autoencoder.fit_generator(generator=train_generator, validation_data=train_generator, 
                              use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                              validation_steps=validation_steps, steps_per_epoch=steps_per_epoch)

    print("saving model...")
    encoder.save("ae/encoder-1.h")
    decoder.save("ae/decoder-1.h")
    print("saved autoencoder.")