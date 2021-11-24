import json
import math
import os
import pickle

import numpy as np

WINDOW_SIZE = 512
GROUP_SIZE = 5  # Group size should be smaller since the skeleton files are shorter
MAX_LEN = WINDOW_SIZE * GROUP_SIZE
COMPILE_TARGET = "XL"  # 'linear', 'XL'
VALID_RATIO = 0.05
VALID_COUNT = 100
VALID_SAMPLING_MODE = "count"
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


def compile(path_root: str):
    # path
    path_indir = os.path.join(path_root, "words")

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
        num_words = len(words)

        if num_words >= MAX_LEN - 2:  # 2 for room
            print(" [!] too long:", num_words)
            continue

        # arrange IO
        x = words[:-1]
        y = words[1:]
        seq_len = len(x)
        print(" > seq_len:", seq_len)

        # pad with eos
        x = np.concatenate([x, np.ones(MAX_LEN - seq_len) * eos_id])
        y = np.concatenate([y, np.ones(MAX_LEN - seq_len) * eos_id])
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

    # split train/test
    if VALID_SAMPLING_MODE == "count":
        valid_count = VALID_COUNT
    elif VALID_SAMPLING_MODE == "ratio":
        valid_count = math.trunc(VALID_RATIO * num_samples)
    else:
        raise ValueError(f"Unknown validation data sampling mode: {VALID_SAMPLING_MODE}")

    test_idx = np.random.choice(num_samples, valid_count, replace=False)
    train_idx = np.setdiff1d(np.arange(num_samples), test_idx)

    # build dictionary for validation data
    valid_dict = {name_list[index.item()]: index.item() for index in test_idx}
    path_valid_dict = os.path.join(path_root, "valid_dict.json")
    with open(path_valid_dict, "w", encoding="utf-8") as file_json:
        json.dump(valid_dict, file_json, indent=4, sort_keys=True, ensure_ascii=False)

    # save train
    path_train = os.path.join(path_root, "train_data_{}.npz".format(COMPILE_TARGET))
    np.savez(
        path_train,
        x=x_final[train_idx],
        y=y_final[train_idx],
        mask=mask_final[train_idx],
        seq_len=np.array(seq_len_list)[train_idx],
        num_groups=np.array(num_groups_list)[train_idx],
    )
    print(" > train x:", x_final[train_idx].shape)

    # save test
    if len(test_idx):
        path_test = os.path.join(path_root, "test_data_{}.npz".format(COMPILE_TARGET))
        np.savez(
            path_test,
            x=x_final[test_idx],
            y=y_final[test_idx],
            mask=mask_final[test_idx],
            seq_len=np.array(seq_len_list)[test_idx],
            num_groups=np.array(num_groups_list)[test_idx],
        )
        print(" >  test x:", x_final[test_idx].shape)
