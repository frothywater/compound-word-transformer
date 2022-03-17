import collections
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


class Saver(object):
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.init_time = time.time()
        self.global_step = 0

        # makedirs
        os.makedirs(exp_dir, exist_ok=True)
        # logging config
        path_logger = os.path.join(exp_dir, "log.txt")
        logging.basicConfig(level=logging.DEBUG, format="%(message)s", filename=path_logger, filemode="w")
        self.logger = logging.getLogger("training monitor")

    def add_summary_msg(self, msg):
        self.logger.debug(msg)

    def add_summary(self, key, val, step=None, cur_time=None):
        if cur_time is None:
            cur_time = time.time() - self.init_time
        if step is None:
            step = self.global_step
        if isinstance(val, float):
            msg_str = f"{key:10s} | {val:.10f} | {step:10d} | {cur_time}"
        else:
            msg_str = f"{key:10s} | {val} | {step:10d} | {cur_time}"
        self.logger.debug(msg_str)

    def save_model(self, model, optimizer=None, dir=None, name="model"):
        if dir is None:
            dir = self.exp_dir
        print(f" [*] saving model : {name}")
        torch.save(model.state_dict(), os.path.join(dir, name + "_params.pt"))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(dir, name + "_opt.pt"))

    def global_step_increment(self):
        self.global_step += 1
