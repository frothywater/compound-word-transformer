import os
import pickle
import random

import numpy as np

WINDOW_SIZE = 512
GROUP_SIZE = 10
MAX_LEN = WINDOW_SIZE * GROUP_SIZE
COMPILE_TARGET = "XL"  # 'linear', 'XL'


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


def shifted_sliding_pair(words: list, offset: int, length: int, pad_word: int, eos_word: int) -> tuple:
    if offset >= len(words):
        raise ValueError("Too short")
    x = np.array(words[offset : offset + length])
    y = np.array(words[offset + 1 : offset + 1 + length])
    valid_length = len(x)
    if offset + 1 + length > len(words):
        # Tail of y exceeds
        y = np.concatenate([y, [eos_word]])
        if offset + length > len(words):
            # Tail of x exceeds
            x = np.concatenate([x, pad_word * np.ones(length - valid_length)])
            y = np.concatenate([y, pad_word * np.ones(length - valid_length)])
    mask = np.concatenate([np.ones(valid_length), np.zeros(length - valid_length)])
    return x, y, mask, valid_length


def get_offsets(length: int, max_length: int, density: int) -> list:
    offset_count = round(density * (length / max_length - 1) + 1)
    if offset_count <= 1:
        return [0]
    left_point_limit = length - max_length
    step = left_point_limit // (offset_count - 1)
    return list(range(0, left_point_limit + 1, step))


def pad_with_repetition(array: list, unit_length: int) -> list:
    result = array.copy()
    if len(array) % unit_length == 0:
        return array
    remain_count = (len(array) // unit_length + 1) * unit_length - len(array)
    extra = random.sample(result, remain_count)
    result += extra
    return result


def compile(path_root: str, mode: str):
    # path
    path_indir = os.path.join(path_root, "words", mode)

    # load dictionary
    path_dictionary = os.path.join(path_root, "dictionary.pkl")
    event2word, word2event = pickle.load(open(path_dictionary, "rb"))
    eos_id = event2word["EOS_None"]
    print(" > eos_id:", eos_id)

    # load all words
    wordfiles = traverse_dir(path_indir, extension=("npy"))
    n_files = len(wordfiles)

    # init
    x_list = []
    y_list = []
    mask_list = []
    seq_len_list = []
    num_groups_list = []
    name_list = []

    # process
    for fidx in range(n_files):
        print("--[{}/{}]-----".format(fidx, n_files))
        file = wordfiles[fidx]
        words = np.load(file)
        word_length = len(words)

        for offset in get_offsets(word_length, WINDOW_SIZE, density=60):
            x, y, mask, valid_length = shifted_sliding_pair(words, offset, WINDOW_SIZE, eos_id, eos_id)
            x_list.append(x)
            y_list.append(y)
            mask_list.append(mask)
            seq_len_list.append(valid_length)
            num_groups_list.append(1)
            name_list.append(file)

    zipped = list(zip(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list))
    padded = pad_with_repetition(zipped, GROUP_SIZE)
    random.shuffle(padded)
    seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list = zip(*padded)

    print("\n\n[Finished]")
    print(" compile target:", COMPILE_TARGET)
    if COMPILE_TARGET == "XL":
        x_final = np.array(x_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        y_final = np.array(y_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        mask_final = np.array(mask_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
    elif COMPILE_TARGET == "linear":
        x_final = np.array(x_list)
        y_final = np.array(y_list)
        mask_final = np.array(mask_list)
    else:
        raise ValueError("Unknown target:", COMPILE_TARGET)
    num_samples = len(seq_len_list)
    print(" >   count:",)
    print(" > x_final:", x_final.shape)
    print(" > y_final:", y_final.shape)
    print(" > mask_final:", mask_final.shape)

    path_data = os.path.join(path_root, f"{mode}_data_{COMPILE_TARGET}.npz")
    np.savez(
        path_data,
        x=x_final,
        y=y_final,
        mask=mask_final,
        seq_len=np.array(seq_len_list),
        num_groups=np.array(num_groups_list),
    )
    print(f" > {mode} x:", x_final.shape)
