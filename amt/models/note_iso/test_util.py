import pretty_midi
import numpy as np

def get_metrics(midi_files):
    # song info:
    # number of songs
    # mean/variance length of songs
    # mean/variance number of instruments per song
    # mean/variance number of notes per song
    # mean/variance note pitch per song
    # mean/variance note length per song
    song_info = {"song_lengths": [], "song_num_instrs": [],
                 "note_pitches": [], "note_lengths": [], "num_notes": []}
    
    # instrument info:
    # number of instruments
    # mean/variance number of notes per instrument
    # mean/variance note pitch per instrument
    # mean/variance note length per instrument
    instr_info = {}
    
    # note info
    # number of notes
    # mean/variance note pitch
    # mean/variance note length
    note_info = {"pitches": [], "lengths": [], "num_notes": 0}
    
    all_instruments = set()
    for song_index, midi_file in enumerate(midi_files):
        pm = pretty_midi.PrettyMIDI(midi_file)
        assert song_index not in song_info
        num_instruments = len(pm.instruments)
        song_length = pm.get_end_time()
        song_info["song_lengths"].append(song_length)
        song_info["song_num_instrs"].append(num_instruments)
        song_info["num_notes"].append(0)
        for instrument in pm.instruments:
            assert instrument.is_drum == False
            all_instruments.add(instrument.program)
            num_notes = len(instrument.notes)
            if instrument.program not in instr_info:
                instr_info[instrument.program] = {"pitches": [], "lengths": [], "num_notes": [], 
                                                  "num_appearances": 0}
            instr_info[instrument.program]["num_notes"].append(num_notes)
            song_info["num_notes"][-1] += num_notes
            note_info["num_notes"] += num_notes
            instr_info[instrument.program]["num_appearances"] += 1
            for note in instrument.notes:
                note_info["pitches"].append(note.pitch)
                note_info["lengths"].append(note.end-note.start)
                instr_info[instrument.program]["pitches"].append(note.pitch)
                instr_info[instrument.program]["lengths"].append(note.end-note.start)
                song_info["note_pitches"].append(note.pitch)
                song_info["note_lengths"].append(note.end-note.start)
                
    data_info = {"num_songs": len(midi_files), 
                 "num_notes": note_info["num_notes"],
                 "all_instruments": all_instruments,
                 "num_instruments": len(all_instruments),
                 
                 "min_note_pitch": np.amin(note_info["pitches"]),
                 "max_note_pitch": np.amax(note_info["pitches"]),
                 "min_note_length": np.amin(note_info["lengths"]),
                 "max_note_length": np.amax(note_info["lengths"]),
                 "mean_note_pitch": np.mean(note_info["pitches"]),
                 "std_note_pitch": np.std(note_info["pitches"]),
                 "mean_note_length": np.mean(note_info["length"]),
                 "std_note_pitch": np.std(note_info["length"]),
                 
                 "mean_song_length": np.mean(song_info["song_lengths"]),
                 "std_song_length": np.std(song_info["song_lengths"]),
                 "mean_song_num_instr": np.mean(song_info["song_num_instrs"]),
                 "std_song_num_instr": np.std(song_info["song_num_instrs"]),
                 "mean_song_note_pitches": np.mean(song_info["note_pitches"]),
                 "std_song_note_pitches": np.std(song_info["note_pitches"]),
                 "mean_song_note_lengths": np.mean(song_info["note_lengths"]),
                 "std_song_note_lengths": np.std(song_info["note_lengths"]),
                 "mean_song_num_notes": np.mean(song_info["num_notes"]),
                 "std_song_num_notes": np.std(song_info["num_notes"])}
    
    del song_info
    
    for instr in instr_info:
        instr_info[instr]["mean_note_pitch"] = np.mean(instr_info[instr]["pitches"])
        instr_info[instr]["std_note_pitch"] = np.std(instr_info[instr]["pitches"])
        instr_info[instr]["mean_note_length"] = np.mean(instr_info[instr]["lengths"])
        instr_info[instr]["std_note_length"] = np.mean(instr_info[instr]["lengths"])
        instr_info[instr]["mean_num_notes"] = np.mean(instr_info[instr]["num_notes"])
        instr_info[instr]["appearance_rate"] = instr_info[instr]["num_appearances"] / len(midi_files)
        del instr_info[instr]["pitches"]
        del instr_info[instr]["lengths"]
        del instr_info[instr]["num_appearances"]
        
    data_info["instr_info"] = instr_info
    
    return data_info
                
                