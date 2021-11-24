import json
import os
import pickle
import re

import numpy as np
import torch
import yaml

from model import TransformerXL

path_root = "data"


def inference_uncond(epoch: int, inference_config):
    os.environ["CUDA_VISIBLE_DEVICES"] = inference_config["gpuID"]

    print("=" * 2, "Inference configs", "=" * 5)
    print(json.dumps(inference_config, indent=1, sort_keys=True))

    # checkpoint information
    train_id = inference_config["train_id"]
    path_checkpoint = os.path.join(path_root, "train", str(train_id))
    midi_folder = os.path.join(path_root, "generated")

    model_path = os.path.join(path_checkpoint, f"ep_{epoch}.pth.tar")
    output_prefix = f"ep_{epoch}_"

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

    # inference
    song_time_list = []
    words_len_list = []
    num_samples = inference_config["num_sample"]
    for idx in range(num_samples):
        print(f"-----{idx}/{num_samples}-----")
        print(midi_folder, output_prefix + str(idx))
        song_time, word_len = model.inference(
            model_path=model_path,
            token_lim=345,
            strategies=["temperature", "nucleus"],
            params={"t": 1.2, "p": 0.9},
            bpm=120,
            output_path="{}/{}.mid".format(midi_folder, output_prefix + str(idx)),
        )

        print("song time:", song_time)
        print("word_len:", word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)

    print("avg token time:", sum(words_len_list) / sum(song_time_list))
    print("avg song time:", np.mean(song_time_list))


def get_all_epochs(path: str, step=10):
    for root, _, files in os.walk(path):
        tar_files = filter(lambda s: s.startswith("ep") and s.endswith("tar"), files)
        digits_regex = re.compile(r"\d+")
        result = [int(digits_regex.findall(name)[0]) for name in tar_files]
        result = [x for x in result if x % step == 0]
        result.sort()
        return result


def main():

    path_config = os.path.join(os.path.dirname(__file__), "config.yml")
    config = yaml.full_load(open(path_config, "r"))
    inference_config = config["INFERENCE"]

    train_id = inference_config["train_id"]
    path_checkpoint = os.path.join(path_root, "train", str(train_id))
    epochs = get_all_epochs(path_checkpoint)

    for epoch in epochs:
        inference_uncond(epoch, inference_config)


if __name__ == "__main__":
    main()
