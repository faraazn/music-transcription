import os
import glob
import mido
import pretty_midi

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
msg_dict = {}
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
for filename in midi_files:
  mid = mido.MidiFile(filename)
  assert mid.type == 1
  is_drums = False
  mid_tracks = mid.tracks[:]
  for track in mid_tracks:
    for idx, msg in enumerate(track):
      if msg.type == 'program_change':
        if msg.channel == 9:  # drums on channel 10
          is_drums = True
      if is_drums and (msg.type == 'note_on' or msg.type == 'note_off') and msg.note == 44:
        identity = msg.type + str(msg.time)
        if identity in msg_dict:
          msg_dict[identity] += 1
        else:
          msg_dict[identity] = 1
    if is_drums:
      break
  if is_drums:
    break
for key in msg_dict:
  print("{}: {}".format(key, msg_dict[key]))
msg_dict = {}
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
for filename in midi_files:
  mid = mido.MidiFile(filename)
  assert mid.type == 1
  mid_tracks = mid.tracks[:]
  for idx, track in enumerate(mid_tracks):
    if idx == 0:
      continue
    for idx, msg in enumerate(track):
      if (msg.type == 'note_on' or msg.type == 'note_off') and msg.note == 57:
        identity = msg.type + str(msg.time)
        if identity in msg_dict:
          msg_dict[identity] += 1
        else:
          msg_dict[identity] = 1
          print(msg)
    break
  break
print("-----")
for key in msg_dict:
  print("{}: {}".format(key, msg_dict[key]))
