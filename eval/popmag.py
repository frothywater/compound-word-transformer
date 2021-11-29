import os
import re
from typing import Pattern

path_dir = "data/generated/popmag"

regex_generated = re.compile(r".*_fromPrompt_\d*\.mid$")
regex_original = re.compile(r".*_raw.mid")


def find(files: list, regex: Pattern[str]):
    return [file for file in files if regex.match(file)]


def main():
    repeats = range(0, 5)
    epochs = range(20, 100 + 1, 20)

    original_songs = []
    found_original_songs = False
    for epoch in epochs:
        epoch_songs: list[str] = []
        for repeat_index in repeats:
            path_inner_dir = os.path.join(path_dir, f"repeat_{repeat_index}", f"epoch_{epoch}")
            files = [os.path.join(path_inner_dir, file) for file in os.listdir(path_inner_dir)]
            epoch_songs += find(files, regex_generated)
            if not found_original_songs:
                original_songs += find(files, regex_original)
                found_original_songs = True
        path_epoch = os.path.join(path_dir, str(epoch))
        for file_path in epoch_songs:
            new_path = os.path.join(path_epoch, os.path.basename(file_path))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(file_path, new_path)

    path_original = os.path.join(path_dir, "original")
    for file_path in original_songs:
        new_path = os.path.join(path_original, os.path.basename(file_path))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(file_path, new_path)


if __name__ == "__main__":
    main()
