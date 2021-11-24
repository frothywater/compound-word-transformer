import re
import json
import os
import pickle
from typing import Callable, Tuple

import numpy as np
import torch
import yaml

from model import TransformerXL

path_root = "data"

calc_valid_loss = False
stop_valid_loss = False


def main():
    # gen config
    modelConfig, trainConfig = get_configs()

    # load dictionary
    event2word, word2event = pickle.load(open(os.path.join(path_root, "dictionary.pkl"), "rb"))

    # load train data
    training_data = np.load(os.path.join(path_root, "train_data_XL.npz"))
    valid_data = np.load(os.path.join(path_root, "test_data_XL.npz")) if calc_valid_loss else None

    device = torch.device(
        "cuda:{}".format(trainConfig["gpuID"]) if not trainConfig["no_cuda"] and torch.cuda.is_available() else "cpu"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = trainConfig["gpuID"]

    print("Device to train:", device)

    resume = trainConfig["resume_training_model"]
    resume_path = get_path_last_model(trainConfig["experiment_dir"]) if resume else None

    # declare model
    model = TransformerXL(modelConfig, device, event2word=event2word, word2event=word2event, is_training=True)

    # train
    model.train(training_data, valid_data, trainConfig, resume_path, stop_valid_loss)


def get_path_last_model(path: str):
    for root, _, files in os.walk(path):
        tar_files = filter(lambda s: s.startswith("ep") and s.endswith("tar"), files)
        digits_regex = re.compile(r"\d+")
        name_epoch_tuples = [(name, int(digits_regex.findall(name)[0])) for name in tar_files]
        name, _ = max(name_epoch_tuples, key=lambda t: t[1])
        return os.path.join(root, name)


def get_configs():
    path_config = os.path.join(os.path.dirname(__file__), "config.yml")
    cfg = yaml.full_load(open(path_config, "r"))

    modelConfig = cfg["MODEL"]
    trainConfig = cfg["TRAIN"]

    train_id = trainConfig["train_id"]
    experiment_dir = os.path.join(path_root, "train", str(train_id))
    if not os.path.exists(experiment_dir):
        print("experiment_dir:", experiment_dir)
        os.makedirs(experiment_dir)
    print("Experiment: ", experiment_dir)
    trainConfig.update({"experiment_dir": experiment_dir})

    with open(os.path.join(experiment_dir, "config.yml"), "w") as f:
        yaml.dump(cfg, f)

    print("=" * 5, "Model configs", "=" * 5)
    print(json.dumps(modelConfig, indent=1, sort_keys=True))
    print("=" * 2, "Training configs", "=" * 5)
    print(json.dumps(trainConfig, indent=1, sort_keys=True))
    return modelConfig, trainConfig


if __name__ == "__main__":
    main()
