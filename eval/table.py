import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    epochs = [int(key) for key in content.keys() if key != "valid"]
    train_loss = [content[str(epoch)]["train_loss"] for epoch in epochs]
    valid_loss = [content[str(epoch)]["valid_loss"] for epoch in epochs]
    raw_data_uncond = metrics_arrays(content["valid"])
    raw_data_cond = metrics_arrays(content["valid"])
    for epoch in epochs:
        raw_data_uncond += metrics_arrays(content[str(epoch)]["uncond"])
        raw_data_cond += metrics_arrays(content[str(epoch)]["cond"])

    data_uncond = np.column_stack(raw_data_uncond)
    data_cond = np.column_stack(raw_data_cond)
    return epochs, data_uncond, data_cond, train_loss, valid_loss


def multi_index(epochs):
    def expand(s: str, array: list):
        return [(s, *item) for item in array]

    inner_1 = [("Mean",), ("STD",)]
    inner_2 = [("KLD",), ("OA",)]
    middle = expand("Intra-set", inner_1) + expand("Inter-set", inner_2)
    result = expand("Validation data", expand("Intra-set", inner_1))
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


def figure_loss(epochs, train_loss, valid_loss):
    fig = plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.legend(loc="lower left")
    plt.close()
    return fig

