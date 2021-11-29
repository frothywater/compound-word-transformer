import json
import os

import numpy as np
from sklearn.model_selection import LeaveOneOut

from loss import get_train_loss, get_valid_loss
from mgeval import core, utils
from utils import get_generated_midi, get_loss, get_test_midi

metrics_shape = {
    "total_used_pitch": (1,),
    "total_pitch_class_histogram": (12,),
    "pitch_class_transition_matrix": (12, 12),
    "pitch_range": (1,),
    "avg_pitch_shift": (1,),
    "total_used_note": (1,),
    "avg_IOI": (1,),
    "note_length_hist": (12,),
    "note_length_transition_matrix": (12, 12),
}
metrics = metrics_shape.keys()


def features(files, num_samples):
    result = {metric: np.zeros((num_samples,) + metrics_shape[metric]) for metric in metrics}
    indices = np.random.choice(len(files), num_samples, replace=False)
    for metric in metrics:
        for i, index in enumerate(indices):
            feature = core.extract_feature(files[index])
            if metric != "total_used_note":
                metric_result = getattr(core.metrics(), metric)(feature)
            else:
                bar_used_notes = core.metrics().bar_used_note(feature, num_bar=32)
                metric_result = np.sum(bar_used_notes)
            result[metric][i] = metric_result
    return result


def cross_valid(set1, set2):
    loo = LeaveOneOut()
    num_samples = len(set1[list(metrics)[0]])
    loo.get_n_splits(np.arange(num_samples))
    result = np.zeros((num_samples, len(metrics), num_samples))
    for i, metric in enumerate(metrics):
        for _, test_index in loo.split(np.arange(num_samples)):
            result[test_index[0]][i] = utils.c_dist(set1[metric][test_index], set2[metric])
    return np.transpose(result, (1, 0, 2)).reshape(len(metrics), -1)


def mean_std(intra):
    mean = {metric: np.mean(intra[i]) for i, metric in enumerate(metrics)}
    std = {metric: np.std(intra[i]) for i, metric in enumerate(metrics)}
    return mean, std


def kl_divergence(intra, inter):
    return {metric: utils.kl_dist(intra[i], inter[i]) for i, metric in enumerate(metrics)}


def overlap_area(intra, inter):
    return {metric: utils.overlap_area(intra[i], inter[i]) for i, metric in enumerate(metrics)}


def metrics_object(gen_midi, features_test, num_samples):
    features_gen = features(gen_midi, num_samples)
    inter = cross_valid(features_gen, features_test)
    intra_gen = cross_valid(features_gen, features_gen)

    mean, std = mean_std(intra_gen)
    kldiv = kl_divergence(intra_gen, inter)
    overlap = overlap_area(intra_gen, inter)

    return {
        "mean": mean,
        "std": std,
        "kldiv": kldiv,
        "overlap": overlap,
    }


def calc_metrics(epochs: range, num_samples: int, path_root: str, path_train: str):
    train_losses = get_train_loss(path_train)
    valid_losses = get_valid_loss(path_train)

    print("Calculating testing midi...")
    test_midi = get_test_midi(path_root)
    features_test = features(test_midi, num_samples)
    intra_test = cross_valid(features_test, features_test)
    mean_test, std_test = mean_std(intra_test)

    result = {"test": {"mean": mean_test, "std": std_test}}

    for epoch in epochs:
        print(f"{epoch=}")
        uncond_midi = get_generated_midi(path_root, epoch, conditional=False)
        cond_midi = get_generated_midi(path_root, epoch, conditional=True)
        train_loss, valid_loss = get_loss(train_losses, valid_losses, epoch)
        result[str(epoch)] = {
            "uncond": metrics_object(uncond_midi, features_test, num_samples),
            "cond": metrics_object(cond_midi, features_test, num_samples),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }

    path_json = os.path.join(path_root, "eval", "metrics.json")
    with open(path_json, "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    path_root = "data"
    path_train = "data/train/1"
    epoch_range = range(20, 100 + 1, 20)

    calc_metrics(epochs=epoch_range, num_samples=100, path_root=path_root, path_train=path_train)
