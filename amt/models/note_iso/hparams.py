import glob
import os

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
wav_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.wav'))
wav_files = [wav_file for wav_file in wav_files]
train_wav_files = wav_files[:len(wav_files)//5]
test_wav_files = wav_files[len(wav_files)//5:]

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
sample_duration = 2 * sample_rate
n_fft = 2048
batch_size = 8
epsilon = 0.00001

epochs = 1
workers = 1
use_multiprocessing = False
validation_steps = 100
steps_per_epoch = None

song_indices = []
instr_indices = []
note_indices = []