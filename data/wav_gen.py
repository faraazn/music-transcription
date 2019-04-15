import scipy.io.wavfile
import glob
import pretty_midi
import os

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
midi_files = [midi_file for midi_file in midi_files]
note_sounds_root_dir = "/home/faraaz/workspace/music-transcription/data/note_sounds/"
sf2_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
num_songs = 500
print_rate = 10

# synthesize midi files
count = 0
midi_files = midi_files[:num_songs]
for midi_file in midi_files:
  if count % print_rate == 0:
    print("{} of {} files synthesized".format(count, num_songs))
  pm = pretty_midi.PrettyMIDI(midi_file=midi_file)
  pm_samples = pm.fluidsynth(fs=sample_rate, sf2_path=sf2_path)
  wav_file = midi_file[:-4] + ".wav"
  scipy.io.wavfile.write(wav_file, sample_rate, pm_samples)
  count += 1
"""
if not os.path.exists(note_sounds_root_dir):
  os.mkdir(note_sounds_root_dir)

# synthesize all types of notes at all pitches
num_programs = 128
velocity = 96  # ff
duration = 1  # seconds
for count, program in enumerate(range(num_programs)):
  if count % print_rate == 0:
    print("{} of {} instruments synthesized".format(count, num_programs))
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program, is_drum=False)
  for pitch in range(21, 109):
    note = pretty_midi.Note(velocity, pitch, 0.0, duration)
    instrument.notes = [note]
    pm.instruments = [instrument]
    pm_samples = pm.fluidsynth(fs=sample_rate, sf2_path=sf2_path)
    wav_file = note_sounds_root_dir + "{}-{}.wav".format(program, pitch)
    scipy.io.wavfile.write(wav_file, sample_rate, pm_samples)
"""
"""
# synthesize all types of drum hits
count = 0
for drum in range(35, 82):
  if count % print_rate == 0:
    print("{} of {} drums synthesized".format(count, 82-35))
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(0, is_drum=True)
  note = pretty_midi.Note(velocity, drum, 0.0, duration)
  instrument.notes = [note]
  pm.instruments = [instrument]
  pm_samples = pm.fluidsynth(fs=sample_rate, sf2_path=sf2_path)
  wav_file = note_sounds_root_dir + "{}{}.wav".format("d", drum)
  scipy.io.wavfile.write(wav_file, sample_rate, pm_samples)
  count += 1
"""
