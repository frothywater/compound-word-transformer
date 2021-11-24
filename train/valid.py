import json
import os
import pickle

import numpy as np
import torch

from model import TransformerXL
from utils import get_all_epochs, get_configs

path_root = "data"


def valid(resume_path: str, model_config, train_config, inference_config):
    # load dictionary
    event2word, word2event = pickle.load(open(os.path.join(path_root, "dictionary.pkl"), "rb"))

    # load train data
    valid_data = np.load(os.path.join(path_root, "test_data_XL.npz"))

    gpuID = inference_config["gpuID"]
    device = torch.device(f"cuda:{gpuID}" if not train_config["no_cuda"] and torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID

    print("Device to validate:", device)

    # declare model
    model = TransformerXL(model_config, device, event2word=event2word, word2event=word2event, is_training=False)

    # validate
    return model.validate_external(valid_data, train_config, resume_path)


def main():
    model_config, train_config, inference_config = get_configs(path_root)
    path_checkpoint = train_config["experiment_dir"]
    epochs = get_all_epochs(path_checkpoint)

    result = []
    for epoch in epochs:
        model_path = os.path.join(path_checkpoint, f"ep_{epoch}.pth.tar")
        val_loss = valid(model_path, model_config, train_config, inference_config)
        result.append((epoch, val_loss))

    path_result = os.path.join(path_checkpoint, "val_loss.json")
    with open(path_result, "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()
