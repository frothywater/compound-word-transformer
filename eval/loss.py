import json
import os

import pandas as pd


def get_train_loss(path_train: str):
    path_csv = os.path.join(path_train, "loss.csv")
    csv_file = pd.read_csv(path_csv)
    array = csv_file.values
    return array[:, 0], array[:, 1]


def get_valid_loss(path_train: str):
    path_json = os.path.join(path_train, "val_loss.json")
    pairs = json.load(open(path_json, "r"))
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    return x, y
