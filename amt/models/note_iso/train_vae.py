import keras
import numpy as np
import glob 
import os
from train_util import NoteIsoSequence
from train_util import get_vae
import time
    
midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
midi_file = next(midi_files)

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
sample_duration = 2 * sample_rate
n_fft = 2048
batch_size = 4
epsilon = 0.00001

epochs = 8
workers = 1
use_multiprocessing = False
validation_steps = 1

song_indices = [0]*64*10
instr_indices = [0]*64*10
note_indices = [0]*64*10
    
if __name__ == "__main__":
    train_generator = NoteIsoSequence([midi_file], sample_duration=sample_duration, 
                                      fs=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)
    vae, my_vae_loss = get_vae()
    vae.summary()
    vae.fit_generator(generator=train_generator, validation_data=train_generator, 
                      use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                      validation_steps=validation_steps)

    print("saving models...")
    vae.save("vae/vae-0.h")
    print("saved vae.")