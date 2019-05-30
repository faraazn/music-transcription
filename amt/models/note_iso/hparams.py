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

train_wav_files = wav_files[0:100]#[:len(wav_files)//5]
test_wav_files = wav_files[100:110]#[len(wav_files)//5:]

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
test_sf2_paths = ["/usr/share/sounds/sf2/arachno.sf2", 
                  "/usr/share/sounds/sf2/timbres_of_heaven.sf2", 
                  "/usr/share/sounds/sf2/MuseScore_General.sf2"]
test_notesounds_paths = ["/home/faraaz/workspace/music-transcription/data/note_sounds/arachno/", 
                         "/home/faraaz/workspace/music-transcription/data/note_sounds/timbres_of_heaven/", 
                         "/home/faraaz/workspace/music-transcription/data/note_sounds/musescore_general/"]
sample_rate = 32000
sample_duration = 1 * sample_rate
n_fft = 128
batch_size = 32
epsilon = 0.0001

epochs = 16
workers = 1
use_multiprocessing = False
validation_steps = 16
steps_per_epoch = None

song_indices = []
instr_indices = []
note_indices = []

encoder = None
decoder = None

EMBED_SIZE = 2
LR = 0.001
F_BINS = n_fft // 2