import json
import os
import pickle
import time

import numpy as np
import torch

from model import TransformerModel
from utils import get_config, set_gpu


def inference():
    # load
    config = get_config()
    path_root = config["path_root"]
    event2word, word2event = pickle.load(open(os.path.join(path_root, "dictionary.pkl"), "rb"))
    path_generated = config["path_generated"]
    os.makedirs(path_generated, exist_ok=True)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(event2word[key]))
    print("num of classes:", n_class)

    # init model
    set_gpu(config["gpu_id"])
    model = TransformerModel(n_class, is_training=False)
    model.cuda()
    model.eval()

    # load model
    load_model = config["load_model"]
    print("[*] load model from:", load_model)
    model.load_state_dict(torch.load(load_model))

    # generate
    while sidx < num_songs:
        start_time = time.time()
        path_outfile = os.path.join(path_generated, )

        res = model.inference_from_scratch(dictionary)
        write_midi(res, path_outfile, word2event)

        word_len = len(res)
        print('song time:', song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)
