import json
import os
import pickle

import numpy as np

TEST_AMOUNT = 50
WINDOW_SIZE = 512
GROUP_SIZE = 2
MAX_LEN = WINDOW_SIZE * GROUP_SIZE
COMPILE_TARGET = "linear"  # 'linear', 'XL'
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
    event2word, word2event = pickle.load(open(path_dictionary, "rb"))

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
        num_words = len(words)
        eos_arr = words[-1][None, ...]

        if num_words >= MAX_LEN - 2:  # 2 for room
            print(" [!] too long:", num_words)
            continue

        # arrange IO
        x = words[:-1].copy()
        y = words[1:].copy()
        seq_len = len(x)
        print(" > seq_len:", seq_len)

        # pad with eos
        pad = np.tile(eos_arr, (MAX_LEN - seq_len, 1))

        x = np.concatenate([x, pad], axis=0)
        y = np.concatenate([y, pad], axis=0)
        mask = np.concatenate([np.ones(seq_len), np.zeros(MAX_LEN - seq_len)])

        # collect
        x_list.append(x)
        y_list.append(y)
        mask_list.append(mask)
        seq_len_list.append(seq_len)
        num_groups_list.append(int(np.ceil(seq_len / WINDOW_SIZE)))
        name_list.append(file)

    # sort by length (descending)
    zipped = zip(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list)
    seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list = zip(*sorted(zipped, key=lambda x: -x[0]))

    print("\n\n[Finished]")
    print(" compile target:", COMPILE_TARGET)
    if COMPILE_TARGET == "XL":
        # reshape
        x_final = np.array(x_list).reshape(len(x_list), GROUP_SIZE, WINDOW_SIZE, -1)
        y_final = np.array(y_list).reshape(len(x_list), GROUP_SIZE, WINDOW_SIZE, -1)
        mask_final = np.array(mask_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
    elif COMPILE_TARGET == "linear":
        x_final = np.array(x_list)
        y_final = np.array(y_list)
        mask_final = np.array(mask_list)
    else:
        raise ValueError("Unknown target:", COMPILE_TARGET)

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
        num_groups=np.array(num_groups_list),
    )
    print(" > {mode} x:", x_final.shape)

