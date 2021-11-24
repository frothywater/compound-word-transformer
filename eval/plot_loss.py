import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

path_exp = "data/train"


def main():
    start_index = None
    if len(sys.argv) > 1:
        start_index = int(sys.argv[1])
    for root, dirs, _ in os.walk(path_exp):
        for dir in dirs:
            path_dir = os.path.join(root, dir)
            plot(path_dir, start_index=start_index)


def plot(path_dir: str, start_index=0):
    path_csv = os.path.join(path_dir, "loss.csv")
    path_png = os.path.join(path_dir, "loss_figure.png")

    csv_file = pd.read_csv(path_csv)
    array = csv_file.values
    plt.xlabel("epoch")
    x = array[start_index:, 0]
    train_loss = array[start_index:, 1]
    plt.plot(x, train_loss, label="train_loss")
    if len(array[0]) > 2:
        valid_loss = array[start_index:, 2]
        plt.plot(x, valid_loss, label="valid_loss")
    plt.legend()
    plt.savefig(path_png)
    plt.close()


if __name__ == "__main__":
    main()
