import os
import pickle
import random

import numpy as np
from model.utils import event_to_word

from dataset.corpus2events import create_pad_event

MAX_LEN = 1024
print("[config] MAX_LEN:", MAX_LEN)

def shifted_sliding_pair(words: list, offset: int, length: int, pad_word) -> tuple:
    if offset >= len(words):
        raise ValueError("Too short")
    x = np.array(words[offset : offset + length])
    y = np.array(words[offset + 1 : offset + 1 + length])
    current_length = len(x)
    mask = np.concatenate([np.ones(current_length), np.zeros(length - current_length)])
    if offset + 1 + length > len(words):
        # Tail of y exceeds
        y = np.concatenate([y, [pad_word]])
        if offset + length > len(words):
            # Tail of x exceeds
            pad = np.tile(pad_word, (length - current_length, 1))
            x = np.concatenate([x, pad])
            y = np.concatenate([y, pad])
    return x, y, mask, current_length


def get_offsets(length: int, max_length: int, density: int, mode: str) -> list:
    if mode == "valid":
        if length <= max_length:
            return [0]
        else:
            return [random.randint(0, length - max_length - 1)]
    offset_count = round(density * (length / max_length - 1) + 1)
    if offset_count <= 1:
        return [0]
    left_point_limit = length - max_length
    step = left_point_limit // (offset_count - 1)
    return list(range(0, left_point_limit + 1, step))


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

    pad_word = event_to_word(create_pad_event(), event2word)

    # process
    for fidx in range(n_files):
        file = wordfiles[fidx]
        words = np.load(file)
        word_length = len(words)

        offsets = get_offsets(word_length, MAX_LEN, density=20, mode=mode)
        for offset in offsets:
            x, y, mask, length = shifted_sliding_pair(words, offset, MAX_LEN, pad_word)
            x_list.append(x)
            y_list.append(y)
            mask_list.append(mask)
            seq_len_list.append(length)

        print(f"[{fidx}/{n_files}] {word_length=} {len(offsets)=}")

    # sort by length (descending)
    zipped = zip(seq_len_list, x_list, y_list, mask_list)
    seq_len_list, x_list, y_list, mask_list = zip(*sorted(zipped, key=lambda x: -x[0]))

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

