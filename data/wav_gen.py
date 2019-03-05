import scipy.io.wavfile
import glob
import pretty_midi
import os

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
sf2_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
sr = 44100

for midi_file in midi_files:
  wav_file = midi_file[:-4] + ".wav"
  print(wav_file)
  fs, samples = scipy.io.wavfile.read(wav_file)
  print("sample rate: {}".format(fs))
  pm = pretty_midi.PrettyMIDI(midi_file=midi_file)
  pm_samples = pm.fluidsynth(fs=fs, sf2_path=sf2_path)
  print("samples {}".format(len(samples)))
  print("pm_samples {}".format(len(pm_samples)))
  break
