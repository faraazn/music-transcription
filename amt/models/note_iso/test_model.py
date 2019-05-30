import keras
import pretty_midi
from train_util import NoteIsoSequence
import sklearn
import time
import glob
import os
import numpy as np
from hparams import *


if __name__ == "__main__":
    encoder = keras.models.load_model("ae/encoder-test2.h")
    
    og_start = time.time()
    start = time.time()
    total_score = 0
    for s_index, test_wav_file in enumerate(test_wav_files):
        test_midi_file = test_wav_file[:-4] + ".mid"
        pm = pretty_midi.PrettyMIDI(test_midi_file)
        sf2_index = np.random.randint(3)
        test_batch_size = 1
        test_generator = NoteIsoSequence([train_wav_files[s_index]], sample_duration=sample_duration, 
                                         sample_rate=sample_rate, n_fft=n_fft, batch_size=test_batch_size,
                                         epsilon=epsilon, song_indices=[], 
                                         instr_indices=[], note_indices=[],
                                         sf2_path=test_sf2_paths[sf2_index],
                                         notesounds_path=test_notesounds_paths[sf2_index])
        test_instr_labels = np.array(test_generator.instr_labels)
        note_embeddings = encoder.predict_generator(test_generator)
        
        # want to be able to guess number of instruments and instrument clusters
        num_instruments = len(pm.instruments)
        kmeans = sklearn.cluster.KMeans(n_clusters=num_instruments)
        cluster_labels = kmeans.fit_predict(note_embeddings)
        
        score = sklearn.metrics.adjusted_rand_score(test_instr_labels, cluster_labels)
        print("{}. {} instruments, score {}, {}s".format(
            s_index, num_instruments, score, time.time()-start))
        start = time.time()
        total_score += score
        
    final_score = total_score / len(test_wav_files)
    print("final score: {}".format(final_score))
    print("{}s".format(time.time()-start))
    
    
    