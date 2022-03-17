import datetime
import os
import pickle
import sys
import time

import numpy as np
import torch

from model import TransformerModel, network_params
from saver import Saver
from utils import get_config, set_gpu


def losses_str(losses):
    losses = [loss.item() for loss in losses]
    tempo, chord, barbeat, type, pitch, duration, velocity = losses
    return f"{tempo=:.3f}, {chord=:.3f}, {barbeat=:.3f}, {type=:.3f}, {pitch=:.3f}, {duration=:.3f}, {velocity=:.3f}"

def train():
    # load
    config = get_config()
    path_root = config["path_root"]
    event2word, _ = pickle.load(open(os.path.join(path_root, "dataset", "dictionary.pkl"), "rb"))
    train_data = np.load(os.path.join(path_root, "dataset", "train.npz"))
    valid_data = np.load(os.path.join(path_root, "dataset", "valid.npz"))

    # create saver
    saver_agent = Saver(os.path.join(path_root, "train", config["train_id"]))

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(event2word[key]))
    print("num of classes:", n_class)

    # init
    set_gpu(config["gpu_id"])
    model = TransformerModel(
        n_token=n_class,
        d_model=config["d_model"],
        d_inner=config["d_inner"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        dropout=config["dropout"],
    )
    model.cuda()
    n_parameters = network_params(model)
    print(f"params amount: {n_parameters}")
    saver_agent.add_summary_msg(f"params amount: {n_parameters}")

    # load model
    load_model = config["load_model"]
    if load_model is not None:
        print("[*] load model from:", load_model)
        model.load_state_dict(torch.load(load_model))

    # optimizers
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["init_lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
    )

    # unpack
    batch_size = config["batch_size"]
    n_epoch = config["n_epoch"]
    max_grad_norm = config["max_grad_norm"]

    train_x = train_data["x"]
    train_y = train_data["y"]
    train_mask = train_data["mask"]
    valid_x = valid_data["x"]
    valid_y = valid_data["y"]
    valid_mask = valid_data["mask"]
    n_batch = len(train_x) // batch_size
    n_batch_valid = len(valid_x) // batch_size

    print("num_batch:", n_batch)
    print("train_x:", train_x.shape)
    print("train_y:", train_y.shape)
    print("train_mask:", train_mask.shape)

    # run
    min_valid_loss = None
    times_valid_loss_increased = 0
    early_stop_patience = 10

    start_time = time.time()
    for epoch in range(n_epoch):
        # train
        train_loss = 0
        train_losses = np.zeros((7,))
        model.train()
        for bidx in range(n_batch):
            saver_agent.global_step_increment()

            # index
            bidx_st = batch_size * bidx
            bidx_ed = batch_size * (bidx + 1)

            # unpack batch data
            batch_x = train_x[bidx_st:bidx_ed]
            batch_y = train_y[bidx_st:bidx_ed]
            batch_mask = train_mask[bidx_st:bidx_ed]

            # to tensor
            batch_x = torch.from_numpy(batch_x).long().cuda()
            batch_y = torch.from_numpy(batch_y).long().cuda()
            batch_mask = torch.from_numpy(batch_mask).float().cuda()

            # run
            losses = model.train_step(batch_x, batch_y, batch_mask)
            loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6]) / 7
            train_loss += loss.item()
            train_losses += np.array([loss.item() for loss in losses])

            # update
            model.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # print
            sys.stdout.write("\x1b[1A\x1b[2K")
            sys.stdout.write(f"epoch: {epoch+1}/{n_epoch}, batch: {bidx}/{n_batch}, loss: {loss.item():.4f}\n")
            sys.stdout.write(f"{losses_str(losses)}\r")
            sys.stdout.flush()

            # log
            saver_agent.add_summary(f"[{epoch+1}] batch loss", loss.item())

        # valid
        valid_loss = 0
        valid_losses = np.zeros((7,))
        model.eval()
        with torch.no_grad():
            for bidx in range(n_batch_valid):
                # index
                bidx_st = batch_size * bidx
                bidx_ed = batch_size * (bidx + 1)

                # unpack batch data
                batch_x = valid_x[bidx_st:bidx_ed]
                batch_y = valid_y[bidx_st:bidx_ed]
                batch_mask = valid_mask[bidx_st:bidx_ed]

                # to tensor
                batch_x = torch.from_numpy(batch_x).long().cuda()
                batch_y = torch.from_numpy(batch_y).long().cuda()
                batch_mask = torch.from_numpy(batch_mask).float().cuda()

                # run
                losses = model.train_step(batch_x, batch_y, batch_mask)
                loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6]) / 7
                valid_loss += loss.item()
                valid_losses += np.array([loss.item() for loss in losses])

                # print
                sys.stdout.write("\x1b[1A\x1b[2K")
                sys.stdout.write(f"valid: {epoch+1}/{n_epoch}, batch: {bidx}/{n_batch_valid}, loss: {loss.item():.4f}\n")
                sys.stdout.write(f"{losses_str(losses)}\r")
                sys.stdout.flush()

                # log
                saver_agent.add_summary(f"[{epoch+1}] valid batch loss", loss.item())

        # epoch loss
        runtime = time.time() - start_time
        train_loss /= n_batch
        train_losses /= n_batch
        valid_loss /= n_batch_valid
        valid_losses /= n_batch_valid
        print(
            f"epoch: {epoch+1}/{n_epoch}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}, time: {datetime.timedelta(seconds=runtime)}"
        )
        print(f"train: {losses_str(train_losses)}")
        print(f"valid: {losses_str(valid_losses)}")
        saver_agent.add_summary(f"[{epoch+1}] train loss", train_loss)
        saver_agent.add_summary(f"[{epoch+1}] valid loss", valid_loss)

        # save model
        saver_agent.save_model(model, name=f"epoch_{epoch+1}")

        # early stop
        if min_valid_loss is None or valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            times_valid_loss_increased = 0
        else:
            times_valid_loss_increased += 1
        if times_valid_loss_increased >= early_stop_patience:
            print("early stopped.")
            break


if __name__ == "__main__":
    train()
