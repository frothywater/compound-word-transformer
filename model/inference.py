import os
import pickle

import numpy as np
import torch

from model import TransformerModel
from utils import get_config, set_gpu, write_midi


def get_test_words(path_root: str):
    path_test = os.path.join(path_root, "words", "test")
    files = [os.path.join(path_test, file) for file in os.listdir(path_test)]
    names = [os.path.basename(file).replace(".mid.pkl.npy", "") for file in files]
    words = [np.load(file) for file in files]
    return list(zip(names, words))


def inference(prompt_bar: int, target_bar: int):
    # load
    config = get_config()
    path_root = config["path_root"]
    event2word, word2event = pickle.load(open(os.path.join(path_root, "dataset", "dictionary.pkl"), "rb"))
    path_generated = os.path.join(path_root, "generated", f"prompt-{prompt_bar}-target-{target_bar}")
    os.makedirs(path_generated, exist_ok=True)

    # config
    n_token = {event_type: len(events) for event_type, events in event2word.items()}
    print(n_token)

    # init model
    set_gpu(config["gpu_id"])
    model = TransformerModel(
        n_token=n_token,
        d_model=config["d_model"],
        d_inner=config["d_inner"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        dropout=config["dropout"],
        is_training=False,
    )
    model.cuda()
    model.eval()

    # load model
    load_model = config["load_model"]
    print("[*] load model from:", load_model)
    model.load_state_dict(torch.load(load_model))

    # generate
    test_words = get_test_words(path_root)
    for i, item in enumerate(test_words):
        name, prompt_words = item
        print(f"[{i+1}/{len(test_words)}] {name}")
        output_path = os.path.join(path_generated, f"{name}_generated.mid")
        original_path = os.path.join(path_generated, f"{name}original.mid")

        generated_words = model.inference(prompt_words, prompt_bar_count=prompt_bar, target_bar_count=target_bar, word2event=word2event)
        write_midi(generated_words, output_path, word2event)
        write_midi(prompt_words, original_path, word2event)


if __name__ == "__main__":
    inference(4, 32)
