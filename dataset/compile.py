import os
import pickle

import numpy as np
from model.utils import event_to_word

from dataset.corpus2events import create_pad_event

MAX_LEN = 1024
print("[config] MAX_LEN:", MAX_LEN)


def traverse_dir(
    root_dir, extension=("mid", "MID"), amount=None, str_=None, is_pure=False, verbose=False, is_sort=False, is_ext=True
):
    if verbose:
        print("[*] Scanning...")
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1 :] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split(".")[-1]
                    pure_path = pure_path[: -(len(ext) + 1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print("Total: %d files" % len(file_list))
        print("Done!!!")
    if is_sort:
        file_list.sort()
    return file_list


def compile(path_root: str, mode: str):
    # path
    path_indir = os.path.join(path_root, "words", mode)

    # load dictionary
    path_dictionary = os.path.join(path_root, "dataset", "dictionary.pkl")
    event2word, _ = pickle.load(open(path_dictionary, "rb"))

    # load all words
    wordfiles = traverse_dir(path_indir, extension=("npy"))
    n_files = len(wordfiles)

    # init
    x_list = []
    y_list = []
    mask_list = []
    seq_len_list = []
    name_list = []

    # process
    for fidx in range(n_files):
        print("--[{}/{}]-----".format(fidx, n_files))
        file = wordfiles[fidx]
        words = np.load(file)
        num_words = len(words)
        pad_word = event_to_word(create_pad_event(), event2word)

        if num_words >= MAX_LEN - 2:  # 2 for room
            print(" [!] too long:", num_words)
            continue

        # arrange IO
        x = words[:-1].copy()
        y = words[1:].copy()
        seq_len = len(x)
        print(" > seq_len:", seq_len)

        # pad
        pad = np.tile(pad_word, (MAX_LEN - seq_len, 1))

        x = np.concatenate([x, pad], axis=0)
        y = np.concatenate([y, pad], axis=0)
        mask = np.concatenate([np.ones(seq_len), np.zeros(MAX_LEN - seq_len)])

        # collect
        x_list.append(x)
        y_list.append(y)
        mask_list.append(mask)
        seq_len_list.append(seq_len)
        name_list.append(file)

    # sort by length (descending)
    zipped = zip(seq_len_list, x_list, y_list, mask_list, name_list)
    seq_len_list, x_list, y_list, mask_list, name_list = zip(*sorted(zipped, key=lambda x: -x[0]))

    print("\n\n[Finished]")
    x_final = np.array(x_list)
    y_final = np.array(y_list)
    mask_final = np.array(mask_list)

    # check
    print(" >   count:",)
    print(" > x_final:", x_final.shape)
    print(" > y_final:", y_final.shape)
    print(" > mask_final:", mask_final.shape)

    # save train
    path_data = os.path.join(path_root, "dataset", f"{mode}.npz")
    np.savez(
        path_data,
        x=x_final,
        y=y_final,
        mask=mask_final,
        seq_len=np.array(seq_len_list),
    )
    print(" > {mode} x:", x_final.shape)

