import keras
import numpy as np
import pretty_midi
import librosa
import glob 
import os
from train_util_e import NoteIsoSequence
from train_util_e import get_encoder, get_decoder, get_discriminator
from train_util_e import program_map, one_hot_map

midi_root_dir = "/home/faraaz/workspace/music-transcription/data/clean_midi/"
midi_files = glob.iglob(os.path.join(midi_root_dir, '**', '*.mid'))
midi_file = next(midi_files)

sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"
sample_rate = 32000
sample_duration = 2 * sample_rate
n_fft = 2048

song_indices = [0]*64*50
instr_indices = [0]*64*50
note_indices = [0]*64*50
train_generator = NoteIsoSequence([midi_file], sample_duration=sample_duration, 
                                  fs=sample_rate, n_fft=n_fft, batch_size=batch_size,
                                  epsilon=epsilon, song_indices=song_indices, 
                                  instr_indices=instr_indices, note_indices=note_indices)

epochs = 8
workers = 1
use_multiprocessing = False
fake_labels = np.zeros((batch_size, 1))
real_labels = np.ones((batch_size, 1))

encoder = get_encoder()
decoder = get_decoder()
discriminator = get_discriminator()

adam = keras.optimizers.Adam(lr=0.01)
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
n_disc_trainable = len(discriminator.trainable_weights)

noisy_spectrogram = keras.layers.Input(shape=(1024, 128, 2))

latent_instr = encoder(noisy_spectrogram)
reconstructed_instr = decoder(latent_instr)

discriminator.trainable = False
predicted_instr = discriminator(reconstructed_instr)

autoencoder = keras.models.Model(noisy_spectrogram, reconstructed_instr)
n_gen_trainable = len(autoencoder.trainable_weights)
autoencoder.compile(optimizer=adam, loss='mean_squared_error')

combined = keras.models.Model(noisy_spectrogram, predicted_instr)
combined.compile(loss='binary_crossentropy', optimizer=adam)
combined.summary()

assert(len(discriminator._collected_trainable_weights) == n_disc_trainable)
assert(len(combined._collected_trainable_weights) == n_gen_trainable)
assert(len(autoencoder._collected_trainable_weights) == n_gen_trainable)

# generator gives X, y each with shape [batch_size, 1024, 128, 2]
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
autoencoder.save("ae-4-a.h")
discriminator.save("disc-4-a.h")
combined.save("comb-4-a.h")
print("saved."))