import os
import pickle
import random

import numpy as np

WINDOW_SIZE = 512
GROUP_SIZE = 5
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


def get_offsets(length: int, max_length: int, count: int) -> list:
    if length <= max_length + count or count <= 1:
        return [0]
    left_point_limit = length - max_length
    step = left_point_limit // (count - 1)
    return list(range(0, left_point_limit + 1, step))


def pad_with_repetition(*arrays, length: int):
    def _pad_with_repetition(array: list, length: int) -> list:
        if len(array) > length:
            raise ValueError("Too long")
        result = array.copy()
        for _ in range(length - len(array)):
            result.append(random.choice(array))
        return result

    zipped = list(zip(*arrays))
    padded = _pad_with_repetition(zipped, length)
    return list(zip(*padded))


def shuffle(*arrays):
    zipped = list(zip(*arrays))
    random.shuffle(zipped)
    shuffled = list(zip(*zipped))
    for i in range(len(arrays)):
        for j in range(len(arrays[i])):
            arrays[i][j] = shuffled[i][j]


def sample(*arrays, k: int):
    zipped = list(zip(*arrays))
    sampled = random.sample(zipped, k)
    return list(zip(*sampled))


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
        file = wordfiles[fidx]
        words = np.load(file)
        word_length = len(words)

        xs, ys, masks, seq_lens = [], [], [], []
        for offset in get_offsets(word_length, WINDOW_SIZE, count=GROUP_SIZE):
            x, y, mask, valid_length = shifted_sliding_pair(words, offset, WINDOW_SIZE, eos_id, eos_id)
            xs.append(x)
            ys.append(y)
            masks.append(mask)
            seq_lens.append(valid_length)
        if len(xs) > GROUP_SIZE:
            xs, ys, masks, seq_lens = sample(xs, ys, masks, seq_lens, k=GROUP_SIZE)
        if len(xs) < GROUP_SIZE:
            xs, ys, masks, seq_lens = pad_with_repetition(xs, ys, masks, seq_lens, length=GROUP_SIZE)
        x_list.append(xs)
        y_list.append(ys)
        mask_list.append(masks)
        seq_len_list.append(seq_lens)
        num_groups_list.append(len(xs))
        name_list.append(file)
        print(f"[{fidx}/{n_files}] {word_length=} group={len(xs)}")

    shuffle(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list)

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
