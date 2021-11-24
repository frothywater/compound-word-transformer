import json
import os
import pickle

import numpy as np
import torch
import yaml

from model import TransformerXL
from utils import get_all_epochs, get_configs

path_root = "data"


def inference_uncond(epoch: int, inference_config):
    os.environ["CUDA_VISIBLE_DEVICES"] = inference_config["gpuID"]

    # checkpoint information
    train_id = inference_config["train_id"]
    path_checkpoint = os.path.join(path_root, "train", str(train_id))
    midi_folder = os.path.join(path_root, "generated", str(epoch))

    model_path = os.path.join(path_checkpoint, f"ep_{epoch}.pth.tar")

    pretrain_config = yaml.full_load(open(os.path.join(path_checkpoint, "config.yml"), "r"))
    model_config = pretrain_config["MODEL"]

    # create result folder
    if not os.path.exists(midi_folder):
        os.mkdir(midi_folder)

    # load dictionary
    event2word, word2event = pickle.load(open(os.path.join(path_root, "dictionary.pkl"), "rb"))

    # declare model
    device = torch.device("cuda" if not inference_config["no_cuda"] and torch.cuda.is_available() else "cpu")
    print("Device to generate:", device)

    # declare model
    model = TransformerXL(model_config, device, event2word=event2word, word2event=word2event, is_training=False)
    _, inner_model = model.get_model(model_path)

    # inference
    song_time_list = []
    words_len_list = []
    num_samples = inference_config["num_sample"]
    for idx in range(num_samples):
        print(f"-----{idx}/{num_samples}-----")
        print(midi_folder, str(idx))
        song_time, word_len = model.inference(
            model=inner_model,
            token_lim=345,
            strategies=["temperature", "top-k"],
            params={"t": 1.5, "k": 10},
            output_path="{}/{}.mid".format(midi_folder, str(idx)),
        )

        print("song time:", song_time)
        print("word_len:", word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)

    print("avg token time:", sum(words_len_list) / sum(song_time_list))
    print("avg song time:", np.mean(song_time_list))


def main():
    _, train_config, inference_config = get_configs(path_root)

    epochs = get_all_epochs(train_config["experiment_dir"])
    for epoch in epochs:
        inference_uncond(epoch, inference_config)


if __name__ == "__main__":
    main()
