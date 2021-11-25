import json
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

path_exp = "data/train"
path_eval = "data/eval"


def get_train_loss(path_dir: str):
    path_csv = os.path.join(path_dir, "loss.csv")
    csv_file = pd.read_csv(path_csv)
    array = csv_file.values
    return array[:, 0], array[:, 1]


def get_valid_loss(path_dir: str):
    path_json = os.path.join(path_dir, "val_loss.json")
    pairs = json.load(open(path_json, "r"))
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    return x, y


def plot(train_id: str, path_eval: str, start_index=0):
    path_dir = os.path.join(path_exp, train_id)
    path_png = os.path.join(path_eval, f"loss_{train_id}.png")

    plt.xlabel("epoch")
    x_train, train_loss = get_train_loss(path_dir)
    x_valid, valid_loss = get_valid_loss(path_dir)
    plt.plot(x_train, train_loss, label="train_loss")
    plt.plot(x_valid, valid_loss, label="valid_loss")

    plt.legend()
    plt.savefig(path_png)
    plt.close()


def main():
    start_index = None
    if len(sys.argv) > 1:
        start_index = int(sys.argv[1])

    os.makedirs(path_eval, exist_ok=True)

    for _, dirs, _ in os.walk(path_exp):
        for dir in dirs:
            plot(train_id=dir, path_eval=path_eval, start_index=start_index)


if __name__ == "__main__":
    main()
