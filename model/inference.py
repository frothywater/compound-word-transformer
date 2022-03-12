import os
import pickle

import numpy as np
import torch
import yaml

from model import TransformerXL
from utils import get_configs

path_root = "data"


def get_test_words(path_root: str):
    path_test = os.path.join(path_root, "words", "test")
    files = [os.path.join(path_test, file) for file in os.listdir(path_test)]
    names = [os.path.basename(file).replace(".mid.pkl.npy", "") for file in files]
    words = [np.load(file) for file in files]
    return zip(range(len(names)), names, words), len(names)


def inference(path_root: str, inference_config, conditional=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = inference_config["gpuID"]

    # checkpoint information
    train_id = inference_config["train_id"]
    epoch = inference_config["epoch"]
    path_checkpoint = os.path.join(path_root, "train", train_id)
    model_path = os.path.join(path_checkpoint, f"ep_{epoch}.pth.tar")
    pretrain_config = yaml.full_load(open(os.path.join(path_checkpoint, "config.yml"), "r"))
    model_config = pretrain_config["MODEL"]

    midi_folder = os.path.join(path_root, "generated", f"{train_id}_{epoch}")
    os.makedirs(midi_folder, exist_ok=True)

    event2word, word2event = pickle.load(open(os.path.join(path_root, "dictionary.pkl"), "rb"))

    # declare model
    device = torch.device("cuda" if not inference_config["no_cuda"] and torch.cuda.is_available() else "cpu")
    print("Device to generate:", device)
    model = TransformerXL(model_config, device, event2word=event2word, word2event=word2event, is_training=False)
    _, inner_model = model.get_model(model_path)

    if conditional:
        test_words, song_count = get_test_words(path_root)
        for i, name, words in test_words:
            print(f"[{i+1}/{song_count}] {midi_folder}/{name}")
            output_path = os.path.join(midi_folder, f"{name}.mid")
            model.inference(
                model=inner_model,
                target_bar_count=32,
                params={"t": 1.5, "k": 10},
                output_path=output_path,
                prompt_words=words,
                prompt_bar_count=1,
            )
    else:
        num_samples = inference_config["num_sample"]
        for idx in range(num_samples):
            print(f"-----{idx}/{num_samples}-----")
            print(midi_folder, str(idx))
            output_path = os.path.join(midi_folder, f"{idx}.mid")
            model.inference(model=inner_model, target_bar_count=32, params={"t": 1.5, "k": 10}, output_path=output_path)


def main():
    _, _, inference_config = get_configs(path_root)

    inference(path_root, inference_config)


if __name__ == "__main__":
    main()
