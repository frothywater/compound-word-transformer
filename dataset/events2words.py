import collections
import json
import os
import pickle

import numpy as np


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


def events2words(path_root: str):
    path_indir = os.path.join(path_root, "events")
    path_outdir = os.path.join(path_root, "words")
    os.makedirs(path_outdir, exist_ok=True)
    os.makedirs(os.path.join(path_root, "dataset"), exist_ok=True)

    # list files
    eventfiles = traverse_dir(path_indir, is_pure=True, is_sort=True, extension=("pkl"))
    n_files = len(eventfiles)
    print("num files:", n_files)

    class_keys = pickle.load(open(os.path.join(path_indir, eventfiles[0]), "rb"))[0].keys()
    print("class keys:", class_keys)

    # --- compile each --- #
    path_dictionary = os.path.join(path_root, "dataset", "dictionary.pkl")
    event2word, word2event = pickle.load(open(path_dictionary, "rb"))
    for fidx in range(len(eventfiles)):
        file = eventfiles[fidx]
        events_list = pickle.load(open(os.path.join(path_indir, file), "rb"))
        fn = os.path.basename(file)
        path_outfile = os.path.join(path_outdir, fn)

        print("({}/{})".format(fidx, len(eventfiles)))
        print(" > from:", file)
        print(" >   to:", path_outfile)

        words = []
        # omg. Please mention the order of elements in word array.
        for eidx, e in enumerate(events_list):
            words_tmp = [event2word[k][e[k]] for k in class_keys]
            words.append(words_tmp)

        # save
        path_outfile = os.path.join(path_outdir, file + ".npy")
        fn = os.path.basename(path_outfile)
        os.makedirs(path_outfile[: -len(fn)], exist_ok=True)
        np.save(path_outfile, words)
