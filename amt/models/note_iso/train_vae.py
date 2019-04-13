from train_util import NoteIsoSequence
from model import get_vae
import time
from hparams import *
    
if __name__ == "__main__":
    print("found {} files".format(len(wav_files)))
    encoder = None
    decoder = None
    
    train_generator = NoteIsoSequence(train_wav_files, sample_duration=sample_duration, 
                                      fs=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)
    test_generator = NoteIsoSequence(test_wav_files, sample_duration=sample_duration, 
                                     fs=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                     epsilon=epsilon, song_indices=song_indices, 
                                     instr_indices=instr_indices, note_indices=note_indices)
    encoder, decoder, vae, my_vae_loss = get_vae(encoder, decoder)
    vae.summary()
    vae.fit_generator(generator=train_generator, validation_data=test_generator, 
                      use_multiprocessing=use_multiprocessing, workers=workers, epochs=epochs, 
                      validation_steps=validation_steps)

    print("saving models...")
    encoder.save("vae/encoder-0.h")
    decoder.save("vae/decoder-0.h")
    print("saved vae.")