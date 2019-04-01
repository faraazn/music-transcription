import keras
import numpy as np
import pretty_midi
import librosa
import glob 
import os
from train_util_d import NoteIsoSequence
from train_util_d import get_model

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
midi_file = next(midi_files)

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
sample_duration = 2 * sample_rate
n_fft = 2048

song_indices = [0]*64*10
instr_indices = [0]*64*10
note_indices = [0]*64*10
train_generator = NoteIsoSequence(
    [midi_file], sample_duration=sample_duration, fs=sample_rate, n_fft=n_fft, 
    mode='train', batch_size=4, song_indices=song_indices, instr_indices=instr_indices,
    note_indices=note_indices)

model = get_model()
model.summary()

model.fit_generator(generator=train_generator, validation_data=train_generator, 
                    use_multiprocessing=True, workers=6, epochs=2, validation_steps=1)

print("saving model...")
model_name = "draft-4-a.h"
model.save(model_name)
print("saved {}".format(model_name))