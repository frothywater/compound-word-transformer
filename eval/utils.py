import json
import os
import re

import pandas as pd


def get_test_midi(path_root: str):
    path_test_dict = os.path.join(path_root, "test_dict.json")
    test_dict: dict = json.load(open(path_test_dict, "r", encoding="utf-8"))
    return [key.replace("words/", "midi/").replace(".pkl.npy", "") for key in test_dict.keys()]


def get_generated_midi(path_root: str, epoch: int, conditional=False):
    folder_name = "cond" if conditional else "uncond"
    path_dir = os.path.join(path_root, "generated", folder_name, str(epoch))
    return [os.path.join(path_dir, file) for file in os.listdir(path_dir) if file.endswith("mid")]


def get_loss_dict(path_train: str):
    path_csv = os.path.join(path_train, "loss.csv")
    content = pd.read_csv(path_csv).values
    result: dict = {}
    for row in content:
        result[row[0]] = {"train_loss": float(row[1]), "valid_loss": float(row[2])}
    return result


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
