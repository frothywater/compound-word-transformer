import json
import os

import metrics
from loss import get_train_loss, get_valid_loss
from utils import get_generated_midi, get_loss, get_valid_midi

path_root = "data"
path_eval = "data/eval"
path_train = "data/train/1"
num_samples = 100

train_losses = get_train_loss(path_train)
valid_losses = get_valid_loss(path_train)


def eval(epoch: int, features_val):
    generated_midi = get_generated_midi(path_root, epoch)

    features_gen = metrics.features(generated_midi, num_samples)
    inter = metrics.cross_valid(features_gen, features_val)
    intra_gen = metrics.cross_valid(features_gen, features_gen)

    mean, std = metrics.mean_std(intra_gen)
    kldiv = metrics.kl_divergence(intra_gen, inter)
    overlap = metrics.overlap_area(intra_gen, inter)
    return mean, std, kldiv, overlap


def main():
    epochs = range(20, 100 + 1, 20)

    train_losses = get_train_loss(path_train)
    valid_losses = get_valid_loss(path_train)

    print("calculating valid midi...")
    valid_midi = get_valid_midi(path_root)
    features_val = metrics.features(valid_midi, num_samples)
    intra_val = metrics.cross_valid(features_val, features_val)
    mean_val, std_val = metrics.mean_std(intra_val)

    result = {"valid": {"mean": mean_val, "std": std_val}}

    for epoch in epochs:
        print(f"{epoch=}")
        mean, std, kldiv, overlap = eval(epoch, features_val)
        train_loss, valid_loss = get_loss(train_losses, valid_losses, epoch)
        result[str(epoch)] = {
            "mean": mean,
            "std": std,
            "kldiv": kldiv,
            "overlap": overlap,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }

    path_json = os.path.join(path_eval, "metrics.json")
    with open(path_json, "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()
