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

    event_dict = {}
    event_dict["type"] = ["Pad", "Metrical", "Note"]
    event_dict["bar-beat"] = [0, "Bar"] + [f"Beat_{value}" for value in beat_range]
    event_dict["tempo"] = [0, "CONTI"] + [f"Tempo_{value}" for value in tempo_range]
    event_dict["pitch"] = [0] + [f"Note_Pitch_{value}" for value in pitch_range]
    event_dict["velocity"] = [0] + [f"Note_Velocity_{value}" for value in velocity_range]
    event_dict["duration"] = [0] + [f"Note_Duration_{value}" for value in duration_range]
    event_dict["chord"] = [0, "CONTI"] + chords

    event2word = {event_type: {event: word for word, event in enumerate(events)} for event_type, events in event_dict.items()}
    word2event = {event_type: {word: event for word, event in enumerate(events)} for event_type, events in event_dict.items()}

    path_dictionary = os.path.join(path_root, "dataset", "dictionary.pkl")
    path_dictionary_json = os.path.join(path_root, "dataset", "dictionary.json")
    pickle.dump((event2word, word2event), open(path_dictionary, "wb"))
    json.dump((event2word, word2event), open(path_dictionary_json, "w"))
