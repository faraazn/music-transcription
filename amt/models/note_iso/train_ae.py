import keras
import numpy as np
import glob 
import os
from train_util import NoteIsoSequence
from train_util import TensorBoardWrapper
from model import get_autoencoder
import time
from hparams import *
from datetime import datetime

if __name__ == "__main__":
    print("found {} files".format(len(wav_files)))
#     encoder = keras.models.load_model("ae/encoder-1.h")
#     decoder = keras.models.load_model("ae/decoder-1.h")
    
    train_generator = NoteIsoSequence(train_wav_files, sample_duration=sample_duration, 
                                      sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)
    test_generator = NoteIsoSequence(test_wav_files, sample_duration=sample_duration, 
                                     sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                     epsilon=epsilon, song_indices=song_indices, 
                                     instr_indices=instr_indices, note_indices=note_indices)
    
    now = datetime.now()
    log_dir = "logs/ae-" + now.strftime("%Y-%m-%d-%H:%M:%S") + "/"
    callbacks = [TensorBoardWrapper(test_generator, log_dir=log_dir, nb_steps=5, histogram_freq=1,
                                    batch_size=batch_size, write_graph=False, write_grads=True)]
    
#     tb = keras.callbacks.TensorBoard(histogram_freq=0, write_grads=True)
    encoder, decoder, autoencoder = get_autoencoder(encoder, decoder)
    autoencoder.summary()

    autoencoder.fit_generator(generator=train_generator, validation_data=train_generator, 
                              use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                              callbacks=callbacks)

    print("saving model...")
    encoder.save("ae/encoder-test.h")
    decoder.save("ae/decoder-test.h")
    print("saved autoencoder.")