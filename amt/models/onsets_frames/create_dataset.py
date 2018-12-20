# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create the recordio files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import glob
import math
import os
import re

import librosa
import numpy as np
import tensorflow as tf
import time

from amt.music import audio_io
from amt.music import midi_io
from amt.music import sequences_lib
from amt.protobuf import music_pb2

FLAGS = tf.app.flags.FLAGS
DATASET = 'clean_midi'
tf.app.flags.DEFINE_string('input_dir', '/home/faraaz/workspace/music-transcription/data/{}/'.format(DATASET),
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', '/home/faraaz/workspace/music-transcription/amt/models/onsets_frames/tfrecord/',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')

test_dirs = ['ENSTDkCl/MUS', 'ENSTDkAm/MUS']
train_dirs = ['AkPnBcht/MUS', 'AkPnBsdf/MUS', 'AkPnCGdD/MUS', 'AkPnStgb/MUS',
              'SptkBGAm/MUS', 'SptkBGCl/MUS', 'StbgTGd2/MUS']

def _find_inactive_ranges(note_sequence):
  """Returns ranges where no notes are active in the note_sequence."""
  start_sequence = sorted(
      note_sequence.notes, key=lambda note: note.start_time, reverse=True)
  end_sequence = sorted(
      note_sequence.notes, key=lambda note: note.end_time, reverse=True)

  notes_active = 0

  time = start_sequence[-1].start_time
  inactive_ranges = []
  if time > 0:
    inactive_ranges.append(0.)
    inactive_ranges.append(time)
  start_sequence.pop()
  notes_active += 1
  # Iterate through all note on events
  while start_sequence or end_sequence:
    if start_sequence and (start_sequence[-1].start_time <
                           end_sequence[-1].end_time):
      if notes_active == 0:
        time = start_sequence[-1].start_time
        inactive_ranges.append(time)
      notes_active += 1
      start_sequence.pop()
    else:
      notes_active -= 1
      if notes_active == 0:
        time = end_sequence[-1].end_time
        inactive_ranges.append(time)
      end_sequence.pop()

  # if the last note is the same time as the end, don't add it
  # remove the start instead of creating a sequence with 0 length
  if inactive_ranges[-1] < note_sequence.total_time:
    inactive_ranges.append(note_sequence.total_time)
  else:
    inactive_ranges.pop()

  assert len(inactive_ranges) % 2 == 0

  inactive_ranges = [(inactive_ranges[2 * i], inactive_ranges[2 * i + 1])
                     for i in range(len(inactive_ranges) // 2)]
  return inactive_ranges


def _last_zero_crossing(samples, start, end):
  """Returns the last zero crossing in the window [start, end)."""
  samples_greater_than_zero = samples[start:end] > 0
  samples_less_than_zero = samples[start:end] < 0
  samples_greater_than_equal_zero = samples[start:end] >= 0
  samples_less_than_equal_zero = samples[start:end] <= 0

  # use np instead of python for loop for speed
  xings = np.logical_or(
      np.logical_and(samples_greater_than_zero[:-1],
                     samples_less_than_equal_zero[1:]),
      np.logical_and(samples_less_than_zero[:-1],
                     samples_greater_than_equal_zero[1:])).nonzero()[0]

  return xings[-1] + start if xings.size > 0 else None


def find_split_points(note_sequence, samples, sample_rate, min_length,
                      max_length):
  """Returns times at which there are no notes.

  The general strategy employed is to first check if there are places in the
  sustained pianoroll where no notes are active within the max_length window;
  if so the middle of the last gap is chosen as the split point.

  If not, then it checks if there are places in the pianoroll without sustain
  where no notes are active and then finds last zero crossing of the wav file
  and chooses that as the split point.

  If neither of those is true, then it chooses the last zero crossing within
  the max_length window as the split point.

  If there are no zero crossings in the entire window, then it basically gives
  up and advances time forward by max_length.

  Args:
      note_sequence: The NoteSequence to split.
      samples: The audio file as samples.
      sample_rate: The sample rate (samples/second) of the audio file.
      min_length: Minimum number of seconds in a split.
      max_length: Maximum number of seconds in a split.

  Returns:
      A list of split points in seconds from the beginning of the file.
  """
  #return [209.71571879166663, 216.14815669166666]
  if not note_sequence.notes:
    return []

  end_time = note_sequence.total_time

  note_sequence_sustain = sequences_lib.apply_sustain_control_changes(
      note_sequence)

  ranges_nosustain = _find_inactive_ranges(note_sequence)
  ranges_sustain = _find_inactive_ranges(note_sequence_sustain)

  nosustain_starts = [x[0] for x in ranges_nosustain]
  sustain_starts = [x[0] for x in ranges_sustain]

  nosustain_ends = [x[1] for x in ranges_nosustain]
  sustain_ends = [x[1] for x in ranges_sustain]

  split_points = [0.]

  while end_time - split_points[-1] > max_length:
    max_advance = split_points[-1] + max_length

    # check for interval in sustained sequence
    pos = bisect.bisect_right(sustain_ends, max_advance)
    if pos < len(sustain_starts) and max_advance > sustain_starts[pos]:
      split_points.append(max_advance)

    # if no interval, or we didn't fit, try the unmodified sequence
    elif pos == 0 or sustain_starts[pos - 1] <= split_points[-1] + min_length:
      # no splits available, use non sustain notes and find close zero crossing
      pos = bisect.bisect_right(nosustain_ends, max_advance)

      if pos < len(nosustain_starts) and max_advance > nosustain_starts[pos]:
        # we fit, great, try to split at a zero crossing
        zxc_start = nosustain_starts[pos]
        zxc_end = max_advance
        last_zero_xing = _last_zero_crossing(
            samples,
            int(math.floor(zxc_start * sample_rate)),
            int(math.ceil(zxc_end * sample_rate)))
        if last_zero_xing:
          last_zero_xing = float(last_zero_xing) / sample_rate
          split_points.append(last_zero_xing)
        else:
          # give up and just return where there are at least no notes
          split_points.append(max_advance)

      else:
        # there are no good places to cut, so just pick the last zero crossing
        # check the entire valid range for zero crossings
        start_sample = int(
            math.ceil((split_points[-1] + min_length) * sample_rate)) + 1
        end_sample = start_sample + (max_length - min_length) * sample_rate
        last_zero_xing = _last_zero_crossing(samples, start_sample, end_sample)

        if last_zero_xing:
          last_zero_xing = float(last_zero_xing) / sample_rate
          split_points.append(last_zero_xing)
        else:
          # give up and advance by max amount
          split_points.append(max_advance)
    else:
      # only advance as far as max_length
      new_time = min(np.mean(ranges_sustain[pos - 1]), max_advance)
      split_points.append(new_time)

  if end_time != split_points[-1] and end_time - split_points[-1] > min_length:
    split_points.append(end_time)

  # ensure that we've generated a valid sequence of splits
  for prev, curr in zip(split_points[:-1], split_points[1:]):
    assert curr > prev
    assert curr - prev <= max_length + 1e-8
    #if curr < end_time:
    assert curr - prev >= min_length - 1e-8
  assert end_time - split_points[-1] < max_length
  return split_points


def filename_to_id(filename):
  """Translate a .wav or .mid path to a MAPS sequence id."""
  return re.match(r'.*MUS-(.*)_[^_]+\.\w{3}',
                  os.path.basename(filename)).group(1)


def generate_train_set(train_file_pairs):
  """Generate the train TFRecord."""
  """train_file_pairs = []
  for directory in train_dirs:i
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      if wav_file not in exclude_files:
        train_file_pairs.append((wav_file, mid_file))
  """
  train_output_name = os.path.join(FLAGS.output_dir,
                                   '{}_train.tfrecord'.format(DATASET))
  count = 0
  with tf.python_io.TFRecordWriter(train_output_name) as writer:
    for pair in train_file_pairs:
      print("{} of {}: {}".format(count, len(train_file_pairs), pair[0]))
      count += 1
      try:
        # load the midi data and convert to a notesequence
        midi_data = tf.gfile.Open(pair[1], 'rb').read()
        ns = midi_io.midi_to_sequence_proto(midi_data)
        
        # load the wav data
        wav_data = tf.gfile.Open(pair[0], 'rb').read()
        samples = audio_io.wav_data_to_samples(wav_data, FLAGS.sample_rate)
        norm_samples = librosa.util.normalize(samples, norm=np.inf)

        splits = find_split_points(ns, norm_samples, FLAGS.sample_rate,
                                   FLAGS.min_length, FLAGS.max_length)

        velocities = [note.velocity for note in ns.notes]
        velocity_max = np.max(velocities)
        velocity_min = np.min(velocities)
        new_velocity_tuple = music_pb2.VelocityRange(
            min=velocity_min, max=velocity_max)
        not_last = True
        for start, end in zip(splits[:-1], splits[1:]):
          new_ns = sequences_lib.extract_subsequence(ns, start, end)
          samples_to_crop = int(start * FLAGS.sample_rate)
          total_samples = int(end * FLAGS.sample_rate)
          new_samples = samples[samples_to_crop:total_samples]
          # check that cut length is valid, ignore if invalid and at the end
          if (end-start)*16000 - len(new_samples) >= 1.0:
            assert not_last
            not_last = False
            continue
          new_wav_data = audio_io.samples_to_wav_data(new_samples, FLAGS.sample_rate)
          example = tf.train.Example(features=tf.train.Features(feature={
              'id':
              tf.train.Feature(bytes_list=tf.train.BytesList(
                  value=[pair[0].encode()]
                  )),
              'sequence':
              tf.train.Feature(bytes_list=tf.train.BytesList(
                  value=[new_ns.SerializeToString()]
                  )),
              'audio':
              tf.train.Feature(bytes_list=tf.train.BytesList(
                  value=[new_wav_data]
                  )),
              'velocity_range':
              tf.train.Feature(bytes_list=tf.train.BytesList(
                  value=[new_velocity_tuple.SerializeToString()]
                  )),
              }))
          writer.write(example.SerializeToString())
      except:
        print("Error occurred.");

def generate_test_set():
  """Generate the test TFRecord."""
  test_file_pairs = []

  """
  for directory in test_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    # path = os.path.join(path, '*.wav')
    wav_files = glob.iglob('src/**/*.wav', recursive=True)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      # all wav files should have corresponding mid files
      mid_file = base_name_root + '.mid'
      test_file_pairs.append((wav_file, mid_file))
  """
  wav_files = glob.iglob('{}/**/*.wav'.format(FLAGS.input_dir), recursive=True)
  for wav_file in wav_files:
    base_name_root, _ = os.path.splitext(wav_file)
    # all wav files should have corresponding mid files
    # timidity replaces . with _, so multiple versions will not exist
    mid_file = base_name_root + '.mid'
    test_file_pairs.append((wav_file, mid_file))
  test_output_name = os.path.join(FLAGS.output_dir,
                                  '{}_test.tfrecord'.format(DATASET))
  total_files = int(len(test_file_pairs) / 20) # lakh midi is about 20x MAPS dataset size
  print('found {} total pairs'.format(total_files))
  #buggy_wav = "/home/faraaz/workspace/music-transcription/data/clean_midi/Gordon Lightfoot/Sundown.wav"
  #buggy_mid = "/home/faraaz/workspace/music-transcription/data/clean_midi/Gordon Lightfoot/Sundown.mid"
  train_file_pairs = test_file_pairs[int(total_files/4):total_files]
  test_file_pairs = test_file_pairs[:int(total_files/4)] # 25% test, 75% train
  train_wavs = [wav for wav, _ in train_file_pairs]
  test_wavs = [wav for wav, _ in test_file_pairs]
  for wav in test_wavs:
    assert wav not in train_wavs
  count = 0
  with tf.python_io.TFRecordWriter(test_output_name) as writer:
    for pair in test_file_pairs:
      print("{} of {}: {}".format(count, len(test_file_pairs), pair[0]))
      count += 1
      try:
        # load the midi data and convert to a notesequence
        midi_data = tf.gfile.Open(pair[1], 'rb').read()
        ns = midi_io.midi_to_sequence_proto(midi_data)

        # load the wav data and resample it.
        samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
        wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)

        velocities = [note.velocity for note in ns.notes]
        velocity_max = np.max(velocities)
        velocity_min = np.min(velocities)
        new_velocity_tuple = music_pb2.VelocityRange(
            min=velocity_min, max=velocity_max)

        example = tf.train.Example(features=tf.train.Features(feature={
            'id':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[pair[0].encode()]
                )),
            'sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[ns.SerializeToString()]
                )),
            'audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[wav_data]
                )),
            'velocity_range':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_velocity_tuple.SerializeToString()]
                )),
            }))
        writer.write(example.SerializeToString())
      except Exception as e:
        print("Error occurred: {}".format(e))
  return train_file_pairs


def main(unused_argv):
  start_time = time.time()
  train_file_pairs = generate_test_set()
  generate_train_set(train_file_pairs)
  print("time elapsed: {}s".format(time.time()-start_time))


def console_entry_point():
  
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
