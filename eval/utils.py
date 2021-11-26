import json
import os
import re


def get_valid_midi(path_root: str):
    path_valid_dict = os.path.join(path_root, "valid_dict.json")
    valid_dict: dict = json.load(open(path_valid_dict, "r", encoding="utf-8"))
    return [key.replace("words/", "midi/").replace(".pkl.npy", "") for key in valid_dict.keys()]


def get_generated_midi(path_root: str, epoch: int):
    path_dir = os.path.join(path_root, "generated", str(epoch))
    return [os.path.join(path_dir, file) for file in os.listdir(path_dir) if file.endswith("mid")]


def get_loss(train_losses, valid_losses, epoch: int):
    for e, loss in zip(train_losses[0], train_losses[1]):
        if e == epoch:
            train_loss = loss
            break
    for e, loss in zip(valid_losses[0], valid_losses[1]):
        if e == epoch:
            valid_loss = loss
            break
    return train_loss, valid_loss


def get_all_epochs(path: str, step=None):
    for _, _, files in os.walk(path):
        tar_files = filter(lambda s: s.startswith("ep") and s.endswith("tar"), files)
        digits_regex = re.compile(r"\d+")
        result = [int(digits_regex.findall(name)[0]) for name in tar_files]
        if step is not None:
            result = [x for x in result if x % step == 0]
        result.sort()
        return result
