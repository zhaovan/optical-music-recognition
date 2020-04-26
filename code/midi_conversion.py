import pretty_midi
import numpy as np

def create_midi(path, notes):
    song = pretty_midi.PrettyMIDI()

    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    current_time = 0
    for (type, length, pitch) in notes:
        length = float(length)
        pitch = float(pitch)
        note_number = int(pitch)*2 + 64
        note_number = min(max(0, note_number), 127)
        
        if (type == b'note'):
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=current_time, end=current_time + length)
            piano.notes.append(note)

        current_time += length

    # Add the piano instrument to the PrettyMIDI object
    song.instruments.append(piano)
    song.write(path + '.mid')