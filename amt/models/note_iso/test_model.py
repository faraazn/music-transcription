import keras
import pretty_midi
from train_util import NoteIsoSequence
import sklearn
import time
import glob
import os
import numpy as np

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
wav_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.wav'))
wav_files = [wav_file for wav_file in wav_files]
num_wav_files = len(wav_files)
train_wav_files = wav_files[:num_wav_files//5]
test_wav_files = wav_files[num_wav_files//5:]
wav_file = wav_files[0]

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
sample_duration = 2 * sample_rate
n_fft = 2048
batch_size = 1
epsilon = 0.00001

epochs = 1
workers = 1
use_multiprocessing = False
validation_steps = 0
steps_per_epoch = None

song_indices = []
instr_indices = []
note_indices = []


if __name__ == "__main__":
    encoder = keras.models.load_model("ae/encoder-0.h")
    
    start = time.time()
    total_score = 0
    for s_index, test_wav_file in enumerate(train_wav_files):
        if s_index % 10 == 9:
            print("{}. score {}".format(s_index, total_score / s_index))
        song_indices = []
        instr_indices = []
        note_indices = []
        test_mid_file = test_wav_file[:-4] + ".mid"
        pm = pretty_midi.PrettyMIDI(test_mid_file)
        for i_index, instrument in enumerate(pm.instruments):
            assert instrument.is_drum == False
            for n_index, note in enumerate(instrument.notes):
                song_indices.append(0)
                instr_indices.append(i_index)
                note_indices.append(n_index)
#         print(len(pm.instruments))
#         print(pm.instruments)
#         print(np.amax(np.array(instr_indices)))
#         print(len(instr_indices))
#         print(pm.get_end_time())
#         print(test_wav_file)
#         break
        assert np.all(np.array(instr_indices) < len(pm.instruments))
                
        test_generator = NoteIsoSequence([train_wav_files[s_index]], sample_duration=sample_duration, 
                                         fs=sample_rate, n_fft=n_fft, batch_size=1,
                                         epsilon=epsilon, song_indices=song_indices, 
                                         instr_indices=instr_indices, note_indices=note_indices)
        note_embeddings = encoder.predict_generator(test_generator)
        
        # want to be able to guess number of instruments and instrument clusters
        num_instruments = len(pm.instruments)
        kmeans = sklearn.cluster.KMeans(n_clusters=num_instruments)
        cluster_labels = kmeans.fit_predict(note_embeddings)
        
        score = sklearn.metrics.adjusted_rand_score(instr_indices, cluster_labels)
        print("score {}".format(score))
        total_score += score
        
    final_score = total_score / len(test_wav_files)
    print("final score: {}".format(final_score))
    print("{}s".format(time.time()-start))
    
    
    