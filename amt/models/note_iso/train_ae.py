import keras
import numpy as np
import glob 
import os
from train_util import NoteIsoSequence
from train_util import TensorBoardWrapper
from model import get_autoencoder, get_autoencoder_plus
import time
from hparams import *
from datetime import datetime

if __name__ == "__main__":
    train_generator = NoteIsoSequence(train_wav_files, sample_duration=sample_duration, 
                                      sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)

    encoder, decoder, classifier, autoencoder = get_autoencoder_plus()
#     encoder = keras.models.load_model("ae+/encoder-1.h")
#     classifier = keras.models.load_model("ae+/classifier-1.h")
    classifier.summary()
#     autoencoder.summary()
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

#     start = time.time()
#     print("training model...")
#     for epoch in range(epochs):
#         for i in range(len(train_generator)):
#             X, y = train_generator[i]
#             onehot_instr, reconstructed_instr = y

#             # classify the instrument
#             c_loss = classifier.train_on_batch(X, onehot_instr)
#             # reconstruct the instrument from latent vector
#             ae_loss = 0 #autoencoder.train_on_batch(X, reconstructed_instr)

#             # option to save generated spectrograms at set intervals 
#         print("epoch {} of {}: c_loss {}, ae_loss {}, {}s".format(
#                epoch, epochs, c_loss, ae_loss, time.time()-start))
#         start = time.time()

    print("saving models...")
    encoder.save("ae+/encoder-5.h")
#     decoder.save("ae+/decoder-1.h")
    classifier.save("ae+/classifier-5.h")
#     autoencoder.save("ae+/autoencoder-1.h")
    print("saved ae+.")
    
    
def train_ae():
    print("found {} files".format(len(wav_files)))
    encoder = keras.models.load_model("ae/encoder-test.h")
    decoder = keras.models.load_model("ae/decoder-test.h")
    
    train_generator = NoteIsoSequence(train_wav_files, sample_duration=sample_duration, 
                                      sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)
    test_generator = NoteIsoSequence(test_wav_files, sample_duration=sample_duration, 
                                     sample_rate=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                     epsilon=epsilon, song_indices=song_indices, 
                                     instr_indices=instr_indices, note_indices=note_indices)
    
    now = datetime.now()
    log_dir = "logs/ae+-" + now.strftime("%Y-%m-%d-%H:%M:%S") + "/"
    callbacks = [TensorBoardWrapper(test_generator, log_dir=log_dir, nb_steps=5, histogram_freq=1,
                                    batch_size=batch_size, write_graph=False, write_grads=True,
                                    write_images=False)]
    
#     tb = keras.callbacks.TensorBoard(histogram_freq=0, write_grads=True)
    encoder, decoder, autoencoder = get_autoencoder(encoder, decoder)
    autoencoder.summary()

    autoencoder.fit_generator(generator=train_generator, validation_data=test_generator, 
                              use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                              callbacks=callbacks)

    print("saving model...")
    encoder.save("ae/encoder-test.h")
    decoder.save("ae/decoder-test.h")
    print("saved autoencoder.")