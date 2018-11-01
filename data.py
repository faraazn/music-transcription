import pretty_midi
import numpy as np
import librosa

tempo = 120.0  # beats per min
input_length = 3.0  # seconds
sr = 22050  # sample rate for synthesis
hop_length = 512
note_velocity = 90
min_pitch = 60  # C4
max_pitch = 69  # A4
num_notes = int(input_length / 60.0 * tempo)
secs_per_beat = 60.0 / tempo
fs = sr / hop_length  # sampling frequency of columns; time between column samples is 1./fs seconds
C1 = 4
C8 = 88
steps_per_song = sr / hop_length * input_length
window_size = 1  # additional frame length added to either side
max_polyphony = 4
p_polyphony = [0.15, 0.35, 0.25, 0.15, 0.1]
assert(steps_per_song / 2 > window_size)

def generate_monophpnic_data(num_songs):
    """Generates a set of num_songs audio inputs and pianoroll labels.

    Args:
        num_songs: number of songs to be generated.

    Returns:
        A tuple (inputs, labels).
    """
    # generate midi

    spectrograms = []  # shape (num_songs, steps_per_song, 84)
    piano_rolls = []  # shape (num_songs, steps_per_song, 84)

    for i in range(num_songs):
        piano_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano_instr = pretty_midi.Instrument(1)
        # play one note per beat
        for j in range(num_notes):
            note_start = j * secs_per_beat
            note_end = (j+1) * secs_per_beat
            note_pitch = np.random.randint(min_pitch, max_pitch+1)
            note = pretty_midi.Note(note_velocity, note_pitch, note_start, note_end)
            piano_instr.notes.append(note)
        piano_midi.instruments.append(piano_instr)

        # generate audio
        piano_audio = piano_midi.fluidsynth(fs=sr)[:int(sr*input_length)]

        # get spectrogram from audio
        C = np.abs(librosa.cqt(piano_audio, sr=sr, hop_length=hop_length, n_bins=84))
        C = C[:, :-1]  # fix off by 1 error

        spectrograms.append(np.transpose(C))

        piano_roll = piano_midi.get_piano_roll(fs=fs)
        # reduce range of piano roll to match spectrogram
        piano_roll = piano_roll[C1:C8, :]
        # convert note velocity to binary note ON or OFF
        piano_roll = (piano_roll != 0).astype(int)

        piano_rolls.append(np.transpose(piano_roll))

    spectrograms = np.array(spectrograms)
    piano_rolls = np.array(piano_rolls)

    return (spectrograms, piano_rolls)

def generate_polyphonic_data(num_songs):
    """Generates a set of num_songs audio inputs and pianoroll labels.

    Args:
        num_songs: number of songs to be generated.

    Returns:
        A tuple (inputs, labels).
    """
    # generate midi

    spectrograms = []  # shape (num_songs, steps_per_song, 84)
    piano_rolls = []  # shape (num_songs, steps_per_song, 84)

    for i in range(num_songs):
        piano_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano_instr = pretty_midi.Instrument(1)
        # play one note per beat
        for j in range(num_notes):
            note_start = j * secs_per_beat
            note_end = (j+1) * secs_per_beat
            polyphony = np.random.choice(max_polyphony+1, p=p_polyphony)
            note_pitches = min_pitch + np.random.choice(max_pitch-min_pitch+1, size=polyphony, replace=False)
            for note_pitch in note_pitches:
                note = pretty_midi.Note(note_velocity, note_pitch, note_start, note_end)
                piano_instr.notes.append(note)
        piano_midi.instruments.append(piano_instr)

        # generate audio
        piano_audio = piano_midi.fluidsynth(fs=sr)[:int(sr*input_length)]

        # get spectrogram from audio
        C = np.abs(librosa.cqt(piano_audio, sr=sr, hop_length=hop_length, n_bins=84))
        C = C[:, :-1]  # fix off by 1 error

        spectrograms.append(np.transpose(C))

        piano_roll = piano_midi.get_piano_roll(fs=fs)
        # reduce range of piano roll to match spectrogram
        piano_roll = piano_roll[C1:C8, :]
        # convert note velocity to binary note ON or OFF
        piano_roll = (piano_roll != 0).astype(int)
        # pad ending silence with 0s
        piano_roll.resize((84, 129))

        piano_rolls.append(np.transpose(piano_roll))

    spectrograms = np.array(spectrograms)
    piano_rolls = np.array(piano_rolls)

    return (spectrograms, piano_rolls)

def generate_samples(num_samples, num_songs, phonic='mono'):
    """Generates a set of num_samples audio frame inputs and pitch labels.

    Args:
        num_samples: number of samples to be generated.

    Returns:
        A tuple (inputs, labels).
    """
    print("generating songs...")
    if phonic == 'mono':
        audio_inputs, pianoroll_labels = generate_monophonic_data(num_songs)
    else:
        audio_inputs, pianoroll_labels = generate_polyphonic_data(num_songs)
    print("done generating songs")

    inputs = []  # shape (num_samples, 1+2*window_size, 84)
    labels = []  # shape (num_samples,)

    print(audio_inputs.shape)
    print(pianoroll_labels.shape)

    for i in range(num_samples):
        rand_song_id = np.random.randint(num_songs)
        rand_song_inputs = audio_inputs[rand_song_id]
        rand_song_labels = pianoroll_labels[rand_song_id]  # shape (steps_per_song, 84)
        # these bounds are not exactly correct
        rand_timestep = np.random.randint(window_size, steps_per_song-window_size)
        rand_input = rand_song_inputs[rand_timestep-window_size:rand_timestep+window_size+1]
        rand_label = rand_song_labels[rand_timestep]

        inputs.append(rand_input)
        labels.append(rand_label)

    inputs = np.array(inputs)
    labels = np.array(labels)

    return inputs, labels