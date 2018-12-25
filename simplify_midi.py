import glob
import os
import pretty_midi
import mido

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"

# remove any generated wav files
"""
wav_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.wav'))
for filename in wav_files:
  os.remove(filename)
"""

# remove files that are repeats (ex. song.1.mid, song.2.mid, ...)
"""
repeat_midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.*.mid'))
for filename in repeat_midi_files:
  os.remove(filename)
"""

# remove files with PrettyMIDI conversion errors
"""
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
for filename in midi_files:
  try:
    pm = pretty_midi.PrettyMIDI(midi_file=filename)
  except IOError:
    print("IOError: {}".format(filename))
    os.remove(filename)
  except KeyError:
    print("KeyError: {}".format(filename))
    os.remove(filename)
  except EOFError:
    print("EOFError: {}".format(filename))
    os.remove(filename)
  except ValueError:
    print("ValueError: {}".format(filename))
    os.remove(filename)
"""

# remove invalid instrument types or instrument class repeats
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
  # sax -> alto saxophone, other reeds -> clarinet
  65: 66, 66: 66, 67: 66, 68: 66, 69: 72, 70: 72, 71: 72, 72: 72,
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
  # percussive -> synth drum
  113: 119, 114: 119, 115: 119, 116: 119, 117: 119, 118: 119, 119: 119, 120: 119,
  # sound effects -> invalid
  121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0
}

midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
for filename in midi_files:
  filename = ''
  pm = pretty_midi.PrettyMIDI(midi_file=filename)
  mid = mido.MidiFile(filename)
  collapsed_programs = set()
  num_programs = 0
  for track in mid.tracks[:]:
    for msg in track:
      if msg.type == 'program_change':
        collapsed_program = program_map[program]
        if collapsed_program in collapsed_programs or \
           collapsed_program == 0 or num_programs >= 3:
          mid.tracks.remove(track)  # delete track
        else:
          collapsed_programs.add(collapsed_program)
          num_programs += 1
        break  # only expect one program_change message per track
  # save simplified midi file back
  mid.save('/home/faraaz/workspace/music-transcription/cleaned.mid')
  break
# finally, limit number of instrument tracks
# and/or limit max number of notes being played at once
