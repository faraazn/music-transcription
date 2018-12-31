import tensorflow as tf
import numpy as np
from amt.protobuf import music_pb2
from amt.music import audio_io
import wave
import six

def wav_to_num_frames(wav_audio, frames_per_second):
  # TODO: make a version using samples, sample rate, hop length
  """Transforms a wav-encoded audio string into number of frames."""
  w = wave.open(six.BytesIO(wav_audio))
  return np.int32(w.getnframes() / w.getframerate() * frames_per_second)


path = "/home/faraaz/workspace/music-transcription/amt/models/onsets_frames/tfrecord/clean_midi_test.tfrecord"
example = tf.train.Example()
sample_rate = 16000
spec_hop_length = 512
frames_per_sec = sample_rate / spec_hop_length
#print(len([record for record in tf.python_io.tf_record_iterator(path)]))
for record in tf.python_io.tf_record_iterator(path):  
  example.ParseFromString(record)
  #print(example)
  f = example.features.feature
  song_name = f['id'].bytes_list.value[0].decode('utf-8')
  wav_bytes = f['wav'].bytes_list.value[0] 
  samples_bytes = f['audio'].float_list.value
  samples_array = np.asarray(samples_bytes, dtype="float32")
  firstsamples = audio_io.load_audio(song_name, 16000)
  wav_data = audio_io.samples_to_wav_data(firstsamples, 16000)
  og_samples_array = audio_io.wav_data_to_samples(wav_data, 16000)
  og_bytes = og_samples_array.tobytes()
  x = np.frombuffer(og_bytes, dtype=np.float32)
  assert wav_data == wav_bytes
  print(wav_to_num_frames(wav_data, frames_per_sec))
  print(len(firstsamples)/512)
  break
