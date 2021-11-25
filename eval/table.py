import json
import os

import numpy as np
import pandas as pd

path_root = "data"
path_eval = "data/eval"

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


def to_array(dict):
    return [dict[metric] for metric in metrics]


def main():
    path_json = os.path.join(path_eval, "metrics.json")
    content = json.load(open(path_json, "r"))

    data = []

    for key in content.keys():
        data.append(to_array(content[key]["mean"]))
        data.append(to_array(content[key]["std"]))

    data_stack = np.column_stack(data)
    df = pd.DataFrame(data_stack)


if __name__ == "__main__":
    main()
