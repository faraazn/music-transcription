import keras
import numpy as np
import pretty_midi
import librosa
import glob 
import os
from train_util_c import NoteIsoSequence
from train_util_c import get_model

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
midi_file = next(midi_files)

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 44100
sample_duration = 44100
n_fft = 1024

train_generator = NoteIsoSequence([midi_file], sample_duration=sample_duration, 
                                  fs=sample_rate, n_fft=n_fft, mode='train',
                                  batch_size=32)

model = get_model()
model.summary()

model.fit_generator(generator=train_generator,
                    validation_data=train_generator,
                    use_multiprocessing=True,
                    workers=6, steps_per_epoch=1000, epochs=1,
                    validation_steps=1)

model.save("draft-3-c.h")