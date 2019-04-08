import keras
import numpy as np
import glob 
import os
from train_util import NoteIsoSequence
from train_util import get_gan
import time

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
midi_file = next(midi_files)

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
sample_duration = 2 * sample_rate
n_fft = 2048
batch_size = 4
epsilon = 0.00001

epochs = 8
workers = 1
use_multiprocessing = False
validation_steps = 0

song_indices = [0]*64*10
instr_indices = [0]*64*10
note_indices = [0]*64*10

if __name__ == "__main__":
    train_generator = NoteIsoSequence([midi_file], sample_duration=sample_duration, 
                                      fs=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                      epsilon=epsilon, song_indices=song_indices, 
                                      instr_indices=instr_indices, note_indices=note_indices)

    autoencoder, discriminator, combined = get_gan()
    combined.summary()

    # generator gives X, y each with shape [batch_size, 1024, 128, 2]
    fake_labels = np.zeros((batch_size, 1))
    real_labels = np.ones((batch_size, 1))
    start = time.time()
    for epoch in range(epochs):
        for i in range(len(train_generator)):
            X, y = train_generator[i]

            # train the discriminator
            fake_instr = autoencoder.predict(X)
            real_instr = y

            d_loss_real = discriminator.train_on_batch(real_instr, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_instr, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # reconstruct the correct output
            r_loss = autoencoder.train_on_batch(X, y)
            # trick the discriminator
            c_loss = combined.train_on_batch(X, real_labels)

            # option to save generated spectrograms at set intervals 
        print("[Dr: {}, {}; Df: {}, {}] [R: {}] [C: {}] {}s".format(
            d_loss_real[0], d_loss_real[1], d_loss_fake[0], d_loss_fake[1], 
            r_loss, c_loss, time.time()-start))
        print(discriminator.get_weights()[0][0][0][0][:5])
        print(combined.get_weights()[0][0][0][0][:5])
        start = time.time()

    print("saving models...")
    autoencoder.save("gan/ae-0.h")
    discriminator.save("gan/disc-0.h")
    combined.save("gan/comb-0.h")
    print("saved gan.")