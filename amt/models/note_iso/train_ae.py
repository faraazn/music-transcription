import keras
import numpy as np
import glob 
import os
from train_util import NoteIsoSequence
from train_util import TensorBoardWrapper
from model import get_autoencoder, get_classifier, get_vae
import time
from hparams import *
from datetime import datetime    
    
def train_classifier():
    train_generator = NoteIsoSequence(train_wav_files, sample_duration=sample_duration, 
                                      sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)

    encoder, classifier = get_classifier()
    classifier.summary()
    
    test_generator = NoteIsoSequence(test_wav_files, sample_duration=sample_duration, 
                                     sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                     epsilon=epsilon, song_indices=song_indices, 
                                     instr_indices=instr_indices, note_indices=note_indices)
    
    now = datetime.now()
    log_dir = "logs/ae+-" + now.strftime("%Y-%m-%d-%H:%M:%S") + "/"
    callbacks = [TensorBoardWrapper(test_generator, log_dir=log_dir, nb_steps=5, histogram_freq=1,
                                    batch_size=batch_size, write_graph=False, write_grads=True,
                                    write_images=False)]
    
    classifier.fit_generator(generator=train_generator, validation_data=test_generator, 
                              use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                              callbacks=callbacks)

    print("saving models...")
    encoder.save("ae+/encoder-5.h")
    classifier.save("ae+/classifier-5.h")
    print("saved ae+.")
    
    
def train_autoencoder():
    print("found {} files".format(len(wav_files)))
    encoder = keras.models.load_model("ae/encoder-test2.h")
    decoder = keras.models.load_model("ae/decoder-test2.h")
    
    train_generator = NoteIsoSequence(train_wav_files, sample_duration=sample_duration, 
                                      sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)
    test_generator = NoteIsoSequence(test_wav_files, sample_duration=sample_duration, 
                                     sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                     epsilon=epsilon, song_indices=song_indices, 
                                     instr_indices=instr_indices, note_indices=note_indices)
    
#     tb = keras.callbacks.TensorBoard(histogram_freq=0, write_grads=True)
    encoder, decoder, autoencoder = get_autoencoder(encoder, decoder)
    autoencoder.summary()
    
    now = datetime.now()
    log_dir = "logs/ae-" + now.strftime("%Y-%m-%d-%H:%M:%S") + "/"
    callbacks = [TensorBoardWrapper(test_generator, log_dir=log_dir, nb_steps=5, histogram_freq=0,
                                    batch_size=batch_size, write_graph=False, write_grads=True,
                                    write_images=False)]

    autoencoder.fit_generator(generator=train_generator, validation_data=test_generator, 
                              use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                              callbacks=callbacks)

    print("saving model...")
    encoder.save("ae/encoder-test2.h")
    decoder.save("ae/decoder-test2.h")
    autoencoder.save("ae/ae-test2.h")
    print("saved autoencoder.")
    

def train_vae():
    print("found {} files".format(len(wav_files)))
#     encoder = keras.models.load_model("vae/encoder-test2.h")
#     decoder = keras.models.load_model("vae/decoder-test2.h")
    
    train_generator = NoteIsoSequence(train_wav_files, sample_duration=sample_duration, 
                                      sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)
    test_generator = NoteIsoSequence(test_wav_files, sample_duration=sample_duration, 
                                     sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                     epsilon=epsilon, song_indices=song_indices, 
                                     instr_indices=instr_indices, note_indices=note_indices)
    
#     tb = keras.callbacks.TensorBoard(histogram_freq=0, write_grads=True)
    encoder, decoder, vae, my_vae_loss = get_vae()
    vae.summary()
    
    now = datetime.now()
    log_dir = "logs/vae-" + now.strftime("%Y-%m-%d-%H:%M:%S") + "/"
    callbacks = [TensorBoardWrapper(test_generator, log_dir=log_dir, nb_steps=5, histogram_freq=0,
                                    batch_size=batch_size, write_graph=False, write_grads=True,
                                    write_images=False)]

    vae.fit_generator(generator=train_generator, validation_data=test_generator, 
                              use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                              callbacks=callbacks)

    print("saving model...")
    encoder.save("vae/encoder-test2.h")
    decoder.save("vae/decoder-test2.h")
    vae.save("vae/vae-test.h")
    print("saved vae.")

    
if __name__ == "__main__":
    train_autoencoder()