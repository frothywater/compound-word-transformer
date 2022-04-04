import json
import os
import pickle

pitch_range = range(36, 84)  # C2-C5
velocity_range = range(40, 128+1, 2)
duration_range = range(0, 16*120+1, 120)
tempo_range = range(32, 224+1, 3)
beat_range = range(0, 16)

note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
chord_qualities = ['M','m','o','+','MM7','Mm7','mM7','mm7','o7','%7','+7','M7','sus']
chords = [f"{root}_{quality}" for root in note_names for quality in chord_qualities]


def build_dictionary(path_root: str):
    os.makedirs(os.path.join(path_root, "dataset"), exist_ok=True)

    events = ["Bar_None", "Pad_None"]
    events += [f"Note_Pitch_{value}" for value in pitch_range]
    events += [f"Note_Velocity_{value}" for value in velocity_range]
    events += [f"Note_Duration_{value}" for value in duration_range]
    events += [f"Chord_{value}" for value in chords]
    events += [f"Tempo_{value}" for value in tempo_range]
    events += [f"Beat_{value}" for value in beat_range]

    event2word = {event: word for word, event in enumerate(events)}
    word2event = {word: event for word, event in enumerate(events)}

    path_dictionary = os.path.join(path_root, "dataset", "dictionary.pkl")
    path_dictionary_json = os.path.join(path_root, "dataset", "dictionary.json")
    pickle.dump((event2word, word2event), open(path_dictionary, "wb"))
    json.dump((event2word, word2event), open(path_dictionary_json, "w"))
