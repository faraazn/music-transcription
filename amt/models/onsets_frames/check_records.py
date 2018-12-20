import tensorflow as tf
import numpy as np
from amt.protobuf import music_pb2
from amt.music import audio_io

path = "/home/faraaz/workspace/music-transcription/amt/models/onsets_frames/tfrecord/clean_midi_train-big.tfrecord"
example = tf.train.Example()
#print(len([record for record in tf.python_io.tf_record_iterator(path)]))
i = 0
for record in tf.python_io.tf_record_iterator(path):
  example.ParseFromString(record)
  f = example.features.feature
  song_name = f['id'].bytes_list.value[0].decode('utf-8')
  #note_seq = f['sequence'].bytes_list.value[0]
  wav_data = f['audio'].bytes_list.value[0]
  #v_range = f['velocity_range'].bytes_list.value[0]
  #vel_range = music_pb2.VelocityRange()
  #vel_range.ParseFromString(v_range)
  y = audio_io.wav_data_to_samples(wav_data, 16000)
  #y = np.fromstring(wav_data)
  i += 1
  if len(y) < 16000*5 or len(y) > 16000*20: # 5 seconds < y < 20 seconds
    print(len(y))
    print(song_name)
    print(type(y))
    print(type(wav_data))
print("total: {}".format(i))
