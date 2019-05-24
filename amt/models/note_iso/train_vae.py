# from train_util import NoteIsoSequence
# from model import get_vae
# import time
# from hparams import *
# import keras

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
    print("found {} files".format(len(wav_files)))
#     encoder = keras.models.load_model("vae/encoder-1.h")
#     decoder = keras.models.load_model("vae/decoder-1.h")
    
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
    
    encoder, decoder, vae, my_vae_loss = get_vae()
    vae.summary()
    vae.fit_generator(generator=train_generator, validation_data=test_generator, 
                      use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                      steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                      callbacks=callbacks)

    print("saving models...")
    encoder.save("vae/encoder-1.h")
    decoder.save("vae/decoder-1.h")
    print("saved vae.")