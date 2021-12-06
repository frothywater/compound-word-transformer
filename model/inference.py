import json
import os
import pickle

import numpy as np
import torch
import yaml

from model import TransformerXL
from utils import get_all_epochs, get_configs

path_root = "data"


def get_test_words(path_root: str):
    path_test_dict = os.path.join(path_root, "test_dict.json")
    test_dict: dict = json.load(open(path_test_dict, "r", encoding="utf-8"))
    paths = [key for key in test_dict.keys()]
    names = [os.path.basename(path).replace(".mid.pkl.npy", "") for path in paths]
    words = [np.load(path) for path in paths]
    return zip(names, words)


def inference(path_root: str, epoch: int, inference_config, conditional=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = inference_config["gpuID"]

    # checkpoint information
    train_id = inference_config["train_id"]
    path_checkpoint = os.path.join(path_root, "train", train_id)
    model_path = os.path.join(path_checkpoint, f"ep_{epoch}.pth.tar")
    pretrain_config = yaml.full_load(open(os.path.join(path_checkpoint, "config.yml"), "r"))
    model_config = pretrain_config["MODEL"]

    folder_name = "cond" if conditional else "uncond"
    midi_folder = os.path.join(path_root, "generated", folder_name, str(epoch))
    os.makedirs(midi_folder, exist_ok=True)

    event2word, word2event = pickle.load(open(os.path.join(path_root, "dictionary.pkl"), "rb"))

    # declare model
    device = torch.device("cuda" if not inference_config["no_cuda"] and torch.cuda.is_available() else "cpu")
    print("Device to generate:", device)
    model = TransformerXL(model_config, device, event2word=event2word, word2event=word2event, is_training=False)
    _, inner_model = model.get_model(model_path)

    if conditional:
        for name, words in get_test_words(path_root):
            print(midi_folder, name)
            output_path = os.path.join(midi_folder, f"{name}.mid")
            model.inference(
                model=inner_model,
                target_bar_count=32,
                params={"t": 1.5, "k": 10},
                output_path=output_path,
                prompt_words=words,
                prompt_bar_count=4,
            )
    else:
        num_samples = inference_config["num_sample"]
        for idx in range(num_samples):
            print(f"-----{idx}/{num_samples}-----")
            print(midi_folder, str(idx))
            output_path = os.path.join(midi_folder, f"{idx}.mid")
            model.inference(model=inner_model, target_bar_count=32, params={"t": 1.5, "k": 10}, output_path=output_path)


def main():
    _, train_config, inference_config = get_configs(path_root)

    epochs = get_all_epochs(train_config["experiment_dir"], step=20)
    for epoch in epochs:
        inference(path_root, epoch, inference_config)
        inference(path_root, epoch, inference_config, conditional=True)


if __name__ == "__main__":
    main()
