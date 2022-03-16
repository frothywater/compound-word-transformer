import os
import pickle

import numpy as np

WINDOW_SIZE = 512
GROUP_SIZE = 5
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


def compile(path_root: str, mode: str):
    # path
    path_indir = os.path.join(path_root, "words", mode)

    # load dictionary
    path_dictionary = os.path.join(path_root, "dictionary.pkl")
    event2word, word2event = pickle.load(open(path_dictionary, "rb"))
    pad_id = event2word["Pad_None"]
    print(" > pad_id:", pad_id)

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

        if word_length >= MAX_LEN - 2:  # 2 for room
            print(" [!] too long:", word_length)
            continue

        words_x, words_y = words[:-1], words[1:]
        x_group, y_group, mask_group = [], [], []
        for offset in range(0, word_length, WINDOW_SIZE):
            x = words_x[offset : offset + WINDOW_SIZE]
            y = words_y[offset : offset + WINDOW_SIZE]
            valid_length = len(x)
            remain_length = WINDOW_SIZE - valid_length
            x = np.concatenate([x, pad_id * np.ones(remain_length)])
            y = np.concatenate([y, pad_id * np.ones(remain_length)])
            mask = np.concatenate([np.ones(valid_length), np.zeros(remain_length)])
            x_group.append(x)
            y_group.append(y)
            mask_group.append(mask)

        group_count = len(x_group)
        if group_count < GROUP_SIZE:
            for _ in range(GROUP_SIZE - group_count):
                x_group.append(np.zeros(WINDOW_SIZE))
                y_group.append(np.zeros(WINDOW_SIZE))
                mask_group.append(np.zeros(WINDOW_SIZE))

        x_list.append(x_group)
        y_list.append(y_group)
        mask_list.append(mask_group)
        seq_len_list.append(word_length)
        num_groups_list.append(group_count)
        name_list.append(file)
        print(f"[{fidx}/{n_files}] {word_length=} {group_count=}")

    # sort by length (descending)
    zipped = zip(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list)
    seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list = zip(*sorted(zipped, key=lambda x: -x[0]))

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
