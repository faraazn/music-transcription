import glob
import os

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
wav_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.wav'))
wav_files = [wav_file for wav_file in wav_files]

# wav_files = []
# maps_root_dir = "/home/faraaz/workspace/music-transcription/data/MAPS/"
# maps_dirs = ["AkPnBcht", "AkPnBsdf", "AkPnCGdD", "AkPnStgb", "ENSTDkAm", 
#              "ENSTDkCl", "SptkBGAm", "SptkBGCl", "StbgTGd2"]
# for maps_dir in maps_dirs:
#     maps_wavs = glob.iglob(os.path.join(maps_root_dir, maps_dir, "MUS", '*.wav'))
#     for maps_wav in maps_wavs:
#         wav_files.append(maps_wav)

train_wav_files = wav_files[:len(wav_files)//5]
test_wav_files = wav_files[len(wav_files)//5:]

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
sample_duration = 1 * sample_rate
n_fft = 2048
batch_size = 16
epsilon = 0.00001

epochs = 10
workers = 1
use_multiprocessing = False
validation_steps = 32
steps_per_epoch = 2048

song_indices = []
instr_indices = []
note_indices = []

encoder = None
decoder = None