import os
import pickle

import numpy as np
import torch

from model import TransformerXL
from utils import get_configs, get_path_last_model

path_root = "data"


def main():
    # gen config
    model_config, train_config, _ = get_configs(path_root)

    # load dictionary
    event2word, word2event = pickle.load(open(os.path.join(path_root, "dictionary.pkl"), "rb"))

    # load train data
    training_data = np.load(os.path.join(path_root, "train_data_XL.npz"))
    valid_data = np.load(os.path.join(path_root, "valid_data_XL.npz"))

    gpuID = train_config["gpuID"]
    device = torch.device(f"cuda:{gpuID}" if not train_config["no_cuda"] and torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID

    print("Device to train:", device)

    resume = train_config["resume_training_model"]
    resume_path = get_path_last_model(train_config["experiment_dir"]) if resume else None

    # declare model
    model = TransformerXL(model_config, device, event2word=event2word, word2event=word2event, is_training=True)

    # train
    model.train(training_data, valid_data, train_config, resume_path)


if __name__ == "__main__":
    main()
