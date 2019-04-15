import keras
import numpy as np
import pretty_midi
import librosa
import glob 
import os
import scipy

class NoteIsoSequence(keras.utils.Sequence):
    """Generate note iso data for training/validation."""
    def __init__(self, song_paths, sample_duration=32000, n_fft=2048, batch_size=16, 
                 sample_rate=32000, sf2_path="/usr/share/sounds/sf2/FluidR3_GM.sf2",
                 instr_indices=[], note_indices=[], song_indices=[], epsilon=0.00001,
                 notesounds_path="/home/faraaz/workspace/music-transcription/data/note_sounds/"):
        self.song_paths = song_paths[:]  # wav files
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sf2_path = sf2_path
        self.sample_duration = sample_duration
        self.n_fft = n_fft
        self.epsilon = epsilon
        self.song_indices = song_indices[:]
        self.instr_indices = instr_indices[:]
        self.note_indices = note_indices[:]
        self.notesounds_path = notesounds_path
        self.indices_given = song_indices and instr_indices and note_indices
        self.on_epoch_end()
        
    def on_epoch_end(self):
        'Update indices after each epoch'
        if not self.indices_given:
            self.song_indices = []
            self.instr_indices = []
            self.note_indices = []
            
            num_songs = len(self.song_paths)
            s_indices = np.random.choice(range(num_songs), size=num_songs, replace=False)
            for s_index in s_indices:
                song_path = self.song_paths[s_index]
                mid_path = song_path[:-4] + ".mid"
                pm = pretty_midi.PrettyMIDI(mid_path)
                num_instr = len(pm.instruments)
                i_indices = np.random.choice(range(num_instr), size=num_instr, replace=False)
                for i_index in i_indices:
                    instrument = pm.instruments[i_index]
                    assert instrument.is_drum == False
                    num_notes = len(instrument.notes)
                    n_indices = np.random.choice(range(num_notes), size=num_notes, replace=False)
                    for n_index in n_indices:
                        self.song_indices.append(s_index)
                        self.instr_indices.append(i_index)
                        self.note_indices.append(n_index)
                # to ensure 1 song per batch, round to nearest batch size
                data_size = len(self.note_indices)
                rounded_size = int(data_size / self.batch_size) * self.batch_size
                self.song_indices = self.song_indices[:rounded_size]
                self.instr_indices = self.instr_indices[:rounded_size]
                self.note_indices = self.note_indices[:rounded_size]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.note_indices) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        song_indices = self.song_indices[self.batch_size*index:self.batch_size*(index+1)]
        instr_indices = self.instr_indices[self.batch_size*index:self.batch_size*(index+1)]
        note_indices = self.note_indices[self.batch_size*index:self.batch_size*(index+1)]
        song_index = song_indices[0]
        assert song_indices.count(song_index) == self.batch_size  # should all be the same
    
        song_path = self.song_paths[song_index]
        mid_path = song_path[:-4] + ".mid"
        pm = pretty_midi.PrettyMIDI(mid_path)
        _, pm_samples = scipy.io.wavfile.read(song_path)
            
        
        X = []
        y = []
        
        for i in range(self.batch_size):
            instr_id = instr_indices[i]
            note_id = note_indices[i]
            instrument = pm.instruments[instr_id]
            note = instrument.notes[note_id]
            sample_start = int(note.start * self.sample_rate)

            padded_samples = self.pad_samples(pm_samples, sample_start)  # shape [sample_duration,]

            noisy_stft = librosa.core.stft(y=padded_samples, n_fft=self.n_fft)  # shape [1025, 65]
            noisy_stft, _, _ = self.if_mel_hp_scale(noisy_stft)  # shape [1024, 65, 2]
            noisy_stft = noisy_stft[:,:63,:]  # shape [1024, 63, 2]
            
            annotation = np.zeros((noisy_stft.shape[0], 1, noisy_stft.shape[2]))  # shape [1024, 1, 2]
            annotation[note.pitch,0,0] = 1  # one hot encode pitch, range (21, 108)
            annotation[0,0,1] = min(self.sample_duration, note.end-note.start) / self.sample_duration  # range (0, 1]
            final_input = np.append(noisy_stft, annotation, axis=1)  # shape [1024, 64, 2]
            
            
            iso_file = self.notesounds_path + "{}-{}.wav".format(instrument.program, note.pitch)
            _, pm_iso_samples = scipy.io.wavfile.read(iso_file)
            pm_iso_samples = self.pad_samples(pm_iso_samples, 0)  # shape [sample_duration,]
            
            iso_stft = librosa.core.stft(y=pm_iso_samples, n_fft=self.n_fft)  # shape [1025, 65]
            iso_stft, _, _ = self.if_mel_hp_scale(iso_stft)  # shape [1024, 65, 2]
            
            iso_pad = np.zeros((iso_stft.shape[0], 1, iso_stft.shape[2]))  # TODO: 0 -> small number?
            final_iso = np.concatenate((iso_stft, iso_pad), axis=1)  # shape (1024, 64, 2)

            assert not np.any(np.isnan(final_iso)) and not np.any(np.isnan(final_input))
            assert np.all(final_iso < 1) and np.all(final_iso > -1)
            assert np.all(final_input <= 1) and np.all(final_input > -1)
            assert final_iso.shape == final_input.shape
            
            X.append(final_input)
            y.append(final_iso)
            
        X = np.array(X)
        y = np.array(y)

        return X, y
    
    
    def pad_samples(self, samples, sample_start):
        padded_samples = samples[:]
        if len(padded_samples > sample_start+self.sample_duration):
            padded_samples = padded_samples[sample_start:sample_start+self.sample_duration]
        if len(padded_samples) < self.sample_duration:
            padded_samples = np.pad(padded_samples, (0, self.sample_duration-len(padded_samples)), 
                                    'constant', constant_values=(0,0))
        return padded_samples
    
    
    def if_mel_hp_scale(self, stft):
        magnitude = np.abs(stft)
        mag_sq = np.square(magnitude)
        lin_mel_matrix = librosa.filters.mel(self.sample_rate, self.n_fft, n_mels=1024)
        mel_sq_mag = np.matmul(lin_mel_matrix, mag_sq)
        log_mel_sq_mag = np.log(mel_sq_mag + self.epsilon)
        mag_scale_factor = max(np.abs(np.amin(log_mel_sq_mag)), np.amax(log_mel_sq_mag)) * 1.25
        final_mag = log_mel_sq_mag / mag_scale_factor
        
        phase = np.angle(stft)
        unwrapped_phase = np.unwrap(phase)
        mel_unwrapped_phase = np.matmul(lin_mel_matrix, unwrapped_phase)
        diff_pad = np.zeros((mel_unwrapped_phase.shape[0], 1))
        mel_inst_freq = np.diff(np.concatenate((diff_pad, mel_unwrapped_phase), axis=1), axis=1)
        phase_scale_factor = max(np.abs(np.amin(mel_inst_freq)), np.amax(mel_inst_freq)) * 1.25
        final_phase = mel_inst_freq / phase_scale_factor
        
        final_stft = np.stack((final_mag, final_phase), axis=2)
        return final_stft, mag_scale_factor, phase_scale_factor
    
    
    def if_hp_scale(self, stft):
        magnitude = np.abs(stft)
        log_mag = np.log(magnitude + self.epsilon)
        mag_scale_factor = max(np.abs(np.amin(log_mag)), np.amax(log_mag)) * 1.25
        final_mag = log_mag / mag_scale_factor
        
        phase = np.angle(stft)
        unwrapped_phase = np.unwrap(phase)
        diff_pad = np.zeros((unwrapped_phase.shape[0], 1))
        inst_freq = np.diff(np.concatenate((diff_pad, unwrapped_phase), axis=1), axis=1)
        phase_scale_factor = max(np.abs(np.amin(inst_freq)), np.amax(inst_freq)) * 1.25
        final_phase = inst_freq / phase_scale_factor
        
        final_stft = np.stack((final_mag, final_phase), axis=2)
        final_stft = final_stft[:-1,:,:]
        return final_stft, mag_scale_factor, phase_scale_factor
    
    
    def hp_scale(self, stft):
        magnitude = np.abs(stft)
        log_mag = np.log(magnitude + self.epsilon)
        mag_scale_factor = max(np.abs(np.amin(log_mag)), np.amax(log_mag)) * 1.25
        scaled_mag = magnitude / mag_scale_factor
        
        phase = np.angle(stft, deg=0)
        phase_scale_factor = np.pi * 1.25
        scaled_phase = phase / phase_scale_factor
        
        final_stft = np.stack((scaled_mag, scaled_phase), axis=2)
        final_stft = final_stft[:-1,:,:]
        return final_stft, mag_scale_factor, phase_scale_factor
    
    
    def if_mel_hp_unscale(self, stft, mag_scale_factor, phase_scale_factor):
        m = librosa.filters.mel(self.sample_rate, self.n_fft, n_mels=1024)
        m_t = np.transpose(m)
        p = np.matmul(m, m_t)
        d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
        mel_lin_matrix = np.matmul(m_t, np.diag(d))
        
        final_mag = stft[:,:,0]
        log_mel_sq_mag = final_mag * mag_scale_factor
        mel_sq_mag = np.exp(log_mel_sq_mag) - self.epsilon
        mag_sq = np.matmul(mel_lin_matrix, mel_sq_mag)
        og_mag = np.sqrt(mag_sq)
        
        final_phase = stft[:,:,1]
        mel_inst_freq = final_phase * phase_scale_factor
        mel_unwrapped_phase = np.cumsum(mel_inst_freq, axis=1)
        unwrapped_phase = np.matmul(mel_lin_matrix, mel_unwrapped_phase)
        
        og_stft = og_mag * np.exp(1j*unwrapped_phase)
        return og_stft
    
    
    def if_hp_unscale(self, stft, mag_scale_factor, phase_scale_factor):
        final_mag = stft[:,:,0]
        log_mag = final_mag * mag_scale_factor
        og_mag = np.exp(log_mag) - self.epsilon
        
        final_phase = stft[:,:,1]
        inst_freq = final_phase * phase_scale_factor
        unwrapped_phase = np.cumsum(inst_freq, axis=1)
        
        og_stft = og_mag * np.exp(1j*unwrapped_phase)
        return og_stft
    
    
    def hp_unscale(self, stft, mag_scale_factor, phase_scale_factor):
        scaled_mag = stft[:,:,0]
        log_mag = scaled_mag * mag_scale_factor
        og_mag = np.exp(log_mag) - self.epsilon
        
        scaled_phase = stft[:,:,1]
        og_phase = scaled_phase * phase_scale_factor
        
        og_stft = og_mag * np.exp(1j*og_phase)
        return og_stft


class TensorBoardWrapper(keras.callbacks.TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        inputs, outputs = self.batch_gen[epoch]
        self.validation_data = [inputs, outputs, np.ones(inputs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)
    
    
program_map = { # map program to invalid id or collapse
  # pianos -> acoustic grand piano 
  1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
  # chromatic percussion -> xylophone
  9: 14, 10: 14, 11: 14, 12: 14, 13: 14, 14: 14, 15: 14, 16: 14,
  # organ, accordion, harmonica -> church organ
  17: 20, 18: 20, 19: 20, 20: 20, 21: 20, 22: 20, 23: 20, 24: 20,
  # guitar -> acoustic guitar (nylon)
  25: 25, 26: 25, 27: 25, 28: 25, 29: 25, 30: 25, 31: 25, 32: 25,
  # bass guitar -> acoustic bass guitar
  33: 33, 34: 33, 35: 33, 36: 33, 37: 33, 38: 33, 39: 33, 40: 33,
  # strings -> violin
  41: 41, 42: 41, 43: 41, 44: 41, 45: 41, 46: 41, 47: 41, 48: 41,
  # ensemble, choir, voice -> invalid
  49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0,
  # brass -> trumpet
  57: 57, 58: 57, 59: 57, 60: 57, 61: 57, 62: 57, 63: 57, 64: 57,
  # reeds -> alto saxophone
  65: 66, 66: 66, 67: 66, 68: 66, 69: 66, 70: 66, 71: 66, 72: 66,
  # pipe -> flute
  73: 74, 74: 74, 75: 74, 76: 74, 77: 74, 78: 74, 79: 74, 80: 74,
  # synth lead -> invalid
  81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0,
  # synth pad -> invalid
  89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0,
  # synth effects -> invalid
  97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0,
  # ethnic -> shamisen
  105: 107, 106: 107, 107: 107, 108: 107, 109: 107, 110: 107, 111: 107, 112: 107,
  # percussive -> invalid
  113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0,
  # sound effects -> invalid
  121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0
}
one_hot_map = {
    1: 1, 14: 2, 20: 3, 25: 4, 33: 5, 41: 6, 57: 7, 66: 8, 74: 9, 107: 10
}