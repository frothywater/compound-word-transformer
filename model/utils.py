import os

import miditoolkit
import numpy as np
import yaml
from dataset.corpus2events import create_bar_event
from miditoolkit.midi.containers import Instrument, Marker, Note, TempoChange


def set_gpu(id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)


def get_config(path_config: str = "model/config.yml"):
    with open(path_config, "r") as config_file:
        return yaml.safe_load(config_file)


def word_to_event(word: list, word2event):
    return {key: word2event[key][word[i]] for i, key in enumerate(word2event.keys())}


def event_to_word(event, event2word):
    return [event2word[key][event[key]] for key in event2word.keys()]


def get_bar_word(event2word):
    return event_to_word(create_bar_event(), event2word)


def crop_words(words: list, bar_length: int, event2word):
    bar_word = get_bar_word(event2word)
    result: list = []
    current_bar_count = 0
    for word in words:
        result.append(word)
        if np.array_equal(word, bar_word):
            current_bar_count += 1
        if current_bar_count > bar_length:
            return result
    return result


def write_midi(words, path_dest, word2event):
    beat_resolution = 480
    bar_resolution = beat_resolution * 4
    tick_resolution = beat_resolution // 4

    class_keys = word2event.keys()
    midi_obj = miditoolkit.midi.parser.MidiFile()
    bar_count, current_pos = 0, 0
    all_notes = []

    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])

        if vals[3] == "Metrical":
            if vals[2] == "Bar":
                bar_count += 1
            elif "Beat" in vals[2]:
                beat_pos = int(vals[2].split("_")[1])
                current_pos = bar_count * bar_resolution + beat_pos * tick_resolution
                # chord
                if vals[1] != "CONTI" and vals[1] != 0:
                    midi_obj.markers.append(Marker(text=str(vals[1]), time=current_pos))
                # tempo
                if vals[0] != "CONTI" and vals[0] != 0:
                    tempo = int(vals[0].split("_")[-1])
                    midi_obj.tempo_changes.append(TempoChange(tempo=tempo, time=current_pos))
            else:
                pass
        elif vals[3] == "Note":
            try:
                pitch = vals[4].split("_")[-1]
                duration = vals[5].split("_")[-1]
                velocity = vals[6].split("_")[-1]
                if int(duration) == 0:
                    duration = 60
                end = current_pos + int(duration)
                all_notes.append(Note(pitch=int(pitch), start=current_pos, end=end, velocity=int(velocity)))
            except:
                continue
        else:
            pass

    piano_track = Instrument(0, is_drum=False, name="piano")
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_dest)
