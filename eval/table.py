import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_loss_dict

metrics = [
    "total_used_pitch",  # PC
    "total_used_note",  # NC
    "total_pitch_class_histogram",  # PCH
    "pitch_class_transition_matrix",  # PCTM
    "pitch_range",  # PR
    "avg_pitch_shift",  # PI
    "avg_IOI",  # IOI
    "note_length_hist",  # NLH
    "note_length_transition_matrix",  # NLTM
]

metric_labels = ["PC", "NC", "PCH", "PCTM", "PR", "PI", "IOI", "NLH", "NLTM"]


def get_metrics_content(path_root: str):
    path_json = os.path.join(path_root, "eval", "metrics.json")
    return json.load(open(path_json, "r"))


def get_data(content):
    def metrics_arrays(dict):
        items = ["mean", "std", "kldiv", "overlap"]
        return [[dict[item][metric] for metric in metrics] for item in items if item in dict]

    epochs = [int(key) for key in content.keys() if key != "test"]
    raw_data_uncond = metrics_arrays(content["test"])
    raw_data_cond = metrics_arrays(content["test"])
    for epoch in epochs:
        raw_data_uncond += metrics_arrays(content[str(epoch)]["uncond"])
        raw_data_cond += metrics_arrays(content[str(epoch)]["cond"])

    data_uncond = np.column_stack(raw_data_uncond)
    data_cond = np.column_stack(raw_data_cond)
    return epochs, data_uncond, data_cond


def multi_index(epochs):
    def expand(s: str, array: list):
        return [(s, *item) for item in array]

    inner_1 = [("Mean",), ("STD",)]
    inner_2 = [("KLD",), ("OA",)]
    middle = expand("Intra-set", inner_1) + expand("Inter-set", inner_2)
    result = expand("Testing data", expand("Intra-set", inner_1))
    for epoch in epochs:
        for item in expand(f"Epoch {epoch}", middle):
            result.append(item)
    return pd.MultiIndex.from_tuples(result)


def dataframe_metrics(data, epochs):
    df = pd.DataFrame(data, index=pd.Index(metric_labels), columns=multi_index(epochs))
    pd.set_option("display.precision", 4)
    return df


def figure_overlap(content, epochs):
    fig = plt.figure(figsize=(16, 12), dpi=80)
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i + 1)
        x = epochs
        y_uncond = [content[str(epoch)]["uncond"]["overlap"][metric] for epoch in epochs]
        y_cond = [content[str(epoch)]["cond"]["overlap"][metric] for epoch in epochs]
        plt.plot(x, y_uncond, label="uncond")
        plt.plot(x, y_cond, label="cond")
        plt.ylim([0.0, 1.0])
        plt.title(metric)
        plt.xlabel("epoch")
        plt.legend()
    plt.close()
    return fig


def figure_loss(path_train: str):
    loss_dict = get_loss_dict(path_train)
    epochs = [int(key) for key in loss_dict]
    train_loss = [value["train_loss"] for value in loss_dict.values()]
    valid_loss = [value["valid_loss"] for value in loss_dict.values()]

    fig = plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.legend(loc="lower left")
    plt.close()
    return fig

