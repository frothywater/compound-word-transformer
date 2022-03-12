import json
import os
import re
import shutil
from random import sample

import mido
import yaml


def get_all_epochs(path: str, step=None, max=None, min=None):
    files = os.listdir(path)
    tar_files = filter(lambda s: s.startswith("ep") and s.endswith("tar"), files)
    digits_regex = re.compile(r"\d+")
    result = [int(digits_regex.findall(name)[0]) for name in tar_files]
    if step:
        result = [x for x in result if x % step == 0]
    if max:
        result = [x for x in result if x <= max]
    if min:
        result = [x for x in result if x >= min]
    result.sort()
    return result


def get_path_last_model(path: str):
    for root, _, files in os.walk(path):
        tar_files = filter(lambda s: s.startswith("ep") and s.endswith("tar"), files)
        digits_regex = re.compile(r"\d+")
        name_epoch_tuples = [(name, int(digits_regex.findall(name)[0])) for name in tar_files]
        name, _ = max(name_epoch_tuples, key=lambda t: t[1])
        return os.path.join(root, name)


def get_configs(path_root: str):
    path_config = os.path.join(os.path.dirname(__file__), "config.yml")
    config = yaml.full_load(open(path_config, "r"))

    model_config = config["MODEL"]
    train_config = config["TRAIN"]
    inference_config = config["INFERENCE"]

    train_id = train_config["train_id"]
    experiment_dir = os.path.join(path_root, "train", train_id)
    print("experiment_dir:", experiment_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    train_config.update({"experiment_dir": experiment_dir})

    with open(os.path.join(experiment_dir, "config.yml"), "w") as f:
        yaml.dump(config, f)

    print("=" * 5, "Model configs", "=" * 5)
    print(json.dumps(model_config, indent=1, sort_keys=True))
    print("=" * 2, "Training configs", "=" * 5)
    print(json.dumps(train_config, indent=1, sort_keys=True))
    print("=" * 2, "Inference configs", "=" * 5)
    print(json.dumps(inference_config, indent=1, sort_keys=True))
    return model_config, train_config, inference_config


def get_bar_length(midi_file: str):
    midi = mido.MidiFile(midi_file)
    return mido.second2tick(midi.length, ticks_per_beat=midi.ticks_per_beat, tempo=500000) / midi.ticks_per_beat / 4


def sample_test_files(valid_path: str, test_path: str, count=50):
    files = os.listdir(valid_path)
    filtered_files = [file for file in files if get_bar_length(os.path.join(valid_path, file)) >= 32]
    sampled_files = sample(filtered_files, k=count)

    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)
    for file in sampled_files:
        print(f"{file} length={get_bar_length(os.path.join(valid_path, file))}")
        shutil.copy(os.path.join(valid_path, file), os.path.join(test_path, file))


if __name__ == "__main__":
    sample_test_files("data/midi/valid", "data/midi/test")
