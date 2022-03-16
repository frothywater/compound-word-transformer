import json
import os
import sys
import time
from typing import List

import miditoolkit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from miditoolkit.midi.containers import Instrument, Marker, Note, TempoChange

import saver
from modules import MemTransformerLM

# ================================ #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {"piano": 0, "melody": 1}


def write_midi(words, path_midi, word2event):
    notes_all = []

    events = [word2event[words[i]] for i in range(len(words))]

    bar_cnt = 0
    cur_beat = 0

    midi_obj = miditoolkit.midi.parser.MidiFile()
    cur_pos = 0

    for i in range(len(events) - 3):
        cur_event = events[i]
        # print(cur_event)
        name = cur_event.split("_")[0]
        attr = cur_event.split("_")
        if name == "Bar":
            bar_cnt += 1
        elif name == "Beat":
            cur_beat = int(attr[1])
            cur_pos = bar_cnt * BAR_RESOL + cur_beat * TICK_RESOL
        elif name == "Chord":
            chord_text = attr[1] + "_" + attr[2]
            midi_obj.markers.append(Marker(text=chord_text, time=cur_pos))
        elif name == "Tempo":
            midi_obj.tempo_changes.append(TempoChange(tempo=int(attr[1]), time=cur_pos))
        else:
            if "Note_Pitch" in events[i] and "Note_Velocity" in events[i + 1] and "Note_Duration" in events[i + 2]:

                pitch = int(events[i].split("_")[-1])
                duration = int(events[i + 2].split("_")[-1])

                if int(duration) == 0:
                    duration = 60

                end = cur_pos + duration
                velocity = int(events[i + 1].split("_")[-1])
                notes_all.append(Note(pitch=pitch, start=cur_pos, end=end, velocity=velocity))

    piano_track = Instrument(0, is_drum=False, name="piano")
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_midi)


# ================================ #
def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class TransformerXL(object):
    def __init__(self, modelConfig, device, event2word, word2event, is_training=True):

        self.event2word = event2word
        self.word2event = word2event
        self.modelConfig = modelConfig

        # model settings
        self.n_layer = modelConfig["n_layer"]
        self.d_model = modelConfig["d_model"]
        self.seq_len = modelConfig["seq_len"]
        self.mem_len = modelConfig["mem_len"]

        self.tgt_len = modelConfig["tgt_len"]
        self.ext_len = modelConfig["ext_len"]
        self.eval_tgt_len = modelConfig["eval_tgt_len"]

        self.init = modelConfig["init"]
        self.init_range = modelConfig["init_range"]
        self.init_std = modelConfig["init_std"]
        self.proj_init_std = modelConfig["proj_init_std"]

        # mode
        self.is_training = is_training
        self.device = device

    def init_weight(self, weight):
        if self.init == "uniform":
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == "normal":
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self.init_weight(m.weight)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find("TransformerLM") != -1:
            if hasattr(m, "r_emb"):
                self.init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self.init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self.init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self.init_bias(m.r_bias)

    def get_model(self, pretrain_model=None):
        n_token = len(self.event2word)
        model = MemTransformerLM(self.modelConfig, n_token, is_training=self.is_training)

        start_epoch = 0
        if pretrain_model:
            checkpoint = torch.load(pretrain_model, map_location="cuda:0")
            print("Pretrained model config:")
            print("epoch: ", checkpoint["epoch"])
            print("best_loss: ", checkpoint["best_loss"])
            print(json.dumps(checkpoint["model_setting"], indent=1, sort_keys=True))
            print(json.dumps(checkpoint["train_setting"], indent=1, sort_keys=True))
            try:
                model.load_state_dict(checkpoint["state_dict"])
                print("{} loaded.".format(pretrain_model))
            except:
                print("Loaded weights have different shapes with the model. Please check your model setting.")
                exit()
            start_epoch = checkpoint["epoch"]
        else:
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init)
        return start_epoch, model.to(self.device)

    def save_checkpoint(self, state, root, save_freq=10, best_val=False):
        if best_val or state["epoch"] % save_freq == 0:
            torch.save(state, os.path.join(root, "ep_{}.pth.tar".format(state["epoch"])))

    def train_loss_record(self, epoch, train_loss, checkpoint_dir, valid_loss=None):
        if valid_loss:
            df = pd.DataFrame(
                {"epoch": [epoch + 1], "train_loss": ["%.3f" % train_loss], "valid_loss": ["%.3f" % valid_loss]}
            )
        else:
            df = pd.DataFrame({"epoch": [epoch + 1], "train_loss": ["%.3f" % train_loss]})

        csv_file = os.path.join(checkpoint_dir, "loss.csv")
        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(os.path.join(checkpoint_dir, "loss.csv"), mode="a", header=False, index=False)

    def note_word_mask(self, data):
        note_words = [word for event, word in self.event2word.items() if event.startswith("Note_Pitch")]
        return np.isin(data, note_words)

    def validate(self, val_data, batch_size, model: MemTransformerLM):
        val_x = val_data["x"]
        val_y = val_data["y"]
        mask = val_data["mask"]
        num_groups = val_data["num_groups"]
        num_batches = len(val_x) // batch_size

        model.eval()
        val_loss = []
        with torch.no_grad():
            for bidx in range(num_batches):
                # index
                bidx_st = batch_size * bidx
                bidx_ed = batch_size * (bidx + 1)

                # get batch
                batch_x = val_x[bidx_st:bidx_ed]
                batch_y = val_y[bidx_st:bidx_ed]
                batch_mask = mask[bidx_st:bidx_ed]
                n_group = np.max(num_groups[bidx_st:bidx_ed])

                # proc groups
                mems: tuple = tuple()
                for gidx in range(n_group):
                    group_x = batch_x[:, gidx, :]
                    group_y = batch_y[:, gidx, :]
                    group_mask = batch_mask[:, gidx, :]

                    group_x = torch.from_numpy(group_x).permute(1, 0).contiguous().to(self.device).long()  # (seq_len, bsz)
                    group_y = torch.from_numpy(group_y).permute(1, 0).contiguous().to(self.device).long()
                    group_mask = torch.from_numpy(group_mask).to(self.device).float()

                    ret = model(group_x, group_y, group_mask, *mems)
                    loss, mems = ret[0], ret[1:]
                    val_loss.append(loss.item())

                    sys.stdout.write(
                        "Validation, batch: {:4d}/{:4d}, group: {:2d}/{:2d} | Loss: {:6f}\r".format(
                            bidx, num_batches, gidx, n_group, val_loss[-1]
                        )
                    )
                    sys.stdout.flush()

        return np.mean(val_loss)

    def validate_external(self, val_data, train_config, resume_path: str):
        batch_size = train_config["batch_size"]
        torch.manual_seed(train_config["seed"])

        # Prepare model
        epoch, model = self.get_model(resume_path)

        print(">>> Start validating")
        st_time = time.time()

        val_loss = self.validate(val_data, batch_size, model)

        epoch_info = f"Epoch: {epoch+1}, Valid Loss: {val_loss:.5f}, T: {time.time() - st_time:.3f}"
        print(epoch_info)

        return val_loss

    def train(self, train_data, valid_data, train_config, resume_path):
        checkpoint_dir = train_config["experiment_dir"]
        batch_size = train_config["batch_size"]
        epoch_count = train_config["num_epochs"]
        # create saver
        saver_agent = saver.Saver(checkpoint_dir)
        # prepare model
        if resume_path:
            start_epoch, model = self.get_model(resume_path)
            print(f"Continue to train from {start_epoch+1} epoch")
        else:
            start_epoch, model = self.get_model()
        # prepare optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config["lr"],
            betas=(train_config["optim_adam_beta1"], train_config["optim_adam_beta2"]),
            weight_decay=train_config["weight_decay"],
        )
        epoch_train_loss = []
        save_freq = train_config["save_freq"]
        n_parameters = network_paras(model)
        print("n_parameters: {:,}".format(n_parameters))
        saver_agent.add_summary_msg(" > params amount: {:,d}".format(n_parameters))

        # unpack training data
        train_x = train_data["x"]
        train_y = train_data["y"]
        mask = train_data["mask"]
        num_groups = train_data["num_groups"]
        num_batches = len(train_x) // batch_size
        # create note masks for pitch shifting
        pitch_shift_mask_x = self.note_word_mask(train_x)
        pitch_shift_mask_y = self.note_word_mask(train_y)
        print(">>> Start training")
        torch.manual_seed(train_config["seed"])

        min_valid_loss = None
        times_valid_loss_increased = 0
        early_stop_patience = 10

        for epoch in range(start_epoch, epoch_count):
            st_time = time.time()
            train_loss = []
            saver_agent.global_step_increment()
            model.train()

            for bidx in range(num_batches):
                model.zero_grad()
                # index
                bidx_st = batch_size * bidx
                bidx_ed = batch_size * (bidx + 1)
                # get batch
                batch_x = train_x[bidx_st:bidx_ed]
                batch_y = train_y[bidx_st:bidx_ed]
                batch_mask = mask[bidx_st:bidx_ed]
                n_group = np.max(num_groups[bidx_st:bidx_ed])
                # process groups
                mems = tuple()
                for gidx in range(n_group):
                    group_x = batch_x[:, gidx, :]
                    group_y = batch_y[:, gidx, :]
                    group_mask = batch_mask[:, gidx, :]
                    group_x = torch.from_numpy(group_x).permute(1, 0).contiguous().to(self.device).long()  # (seq_len, bsz)
                    group_y = torch.from_numpy(group_y).permute(1, 0).contiguous().to(self.device).long()
                    group_mask = torch.from_numpy(group_mask).to(self.device).float()

                    ret = model(group_x, group_y, group_mask, *mems)
                    loss, mems = ret[0], ret[1:]
                    train_loss.append(loss.item())
                    loss.backward()

                    sys.stdout.write(
                        f"epoch: {epoch+1}/{epoch_count}, batch: {bidx+1}/{num_batches}, "
                        + f"group: {gidx+1}/{n_group} | loss: {loss.item():5f}\r"
                    )
                    sys.stdout.flush()
                optimizer.step()

            # validate
            if valid_data:
                valid_loss = self.validate(valid_data, batch_size, model)
                saver_agent.add_summary("valid loss", valid_loss)

            average_train_loss = sum(train_loss) / len(train_loss)
            saver_agent.add_summary("epoch loss", average_train_loss)

            epoch_train_loss.append(average_train_loss)
            epoch_info = f"Epoch: {epoch+1}, Train Loss: {average_train_loss:.5f}"
            if valid_data:
                epoch_info += f", Valid Loss: {valid_loss:.5f}"
            epoch_info += f", Time: {time.time() - st_time:.3f}"
            print(epoch_info)

            self.train_loss_record(epoch, average_train_loss, checkpoint_dir, valid_loss)
            state = {
                "epoch": epoch + 1,
                "model_setting": self.modelConfig,
                "train_setting": train_config,
                "state_dict": model.state_dict(),
                "best_loss": average_train_loss,
                "optimizer": optimizer.state_dict(),
            }
            self.save_checkpoint(state, checkpoint_dir, save_freq)

            if average_train_loss < 0.01:
                print("Experiment [{}] finished at loss < 0.01.".format(checkpoint_dir))
                break

            # Early stopping
            if min_valid_loss is None or valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                times_valid_loss_increased = 0
            else:
                times_valid_loss_increased += 1
            if times_valid_loss_increased >= early_stop_patience:
                print("Early stopped.")
                break

    def inference(
        self,
        model: MemTransformerLM,
        target_bar_count: int,
        params: dict,
        output_path: str,
        prompt_words: list,
        prompt_bar_count: int,
    ):
        batch_size = 1
        mems: tuple = tuple()
        bar_none_word = self.event2word["Bar_None"]
        start_time = time.time()
        model.eval()

        if prompt_words is not None:
            # Conditional
            # Select words within the given bar range
            current_bar = 0
            prompt_words_cropped = []
            for prompt_word in prompt_words:
                if prompt_word == bar_none_word:
                    current_bar += 1
                if current_bar > prompt_bar_count:
                    break
                prompt_words_cropped.append(prompt_word)

            # Feed the prompt (until before the last word), but leave out any outputs
            with torch.no_grad():
                for prompt_word in prompt_words_cropped[:-1]:
                    temp_x_teaching = np.zeros((1, batch_size))
                    temp_x_teaching[0][0] = prompt_word
                    x = torch.from_numpy(temp_x_teaching).long().to(self.device)
                    _, mems = model.generate(x, *mems)

            bar_count = prompt_bar_count
            generated_words = []
            temp_x = np.zeros((1, batch_size))
            temp_x[0][0] = prompt_words_cropped[-1]  # Feed the last word in the real inference stage
        else:
            # Unconditional
            bar_count = 0
            generated_words = [bar_none_word]
            temp_x = np.zeros((1, batch_size))
            temp_x[0][0] = bar_none_word

        # With the memory, generate the real part
        while bar_count < target_bar_count:
            # Feed in `temp_x`
            with torch.no_grad():
                x = torch.from_numpy(temp_x).long().to(self.device)
                _logits, mems = model.generate(x, *mems)
                logits = _logits.cpu().squeeze().detach().numpy()

            # Sample with temperature, and optional strategy (top-k or nucleus)
            temperature = params["t"] if "t" in params else 1.0
            probs = self.temperature(logits, temperature)
            if "k" in params:
                word = self.topk(probs=probs, k=params["k"])
            elif "p" in params:
                word = self.nucleus(probs=probs, p=params["p"])
            generated_words.append(word)

            # Set `temp_x` as the last generated word
            temp_x = np.zeros((1, batch_size))
            temp_x[0][0] = word

            if word == self.event2word["Bar_None"]:
                bar_count += 1
            sys.stdout.write(f"{self.word2event[word]}, {len(generated_words)=}, {bar_count=}\r")
            sys.stdout.flush()

        # Write midi files
        if prompt_words is not None:
            original_output_path = output_path.replace(".mid", "_original.mid")
            generated_output_path = output_path.replace(".mid", "_generated.mid")
            write_midi(prompt_words, original_output_path, self.word2event)
            write_midi(prompt_words_cropped + generated_words, generated_output_path, self.word2event)
        else:
            write_midi(generated_words, output_path, self.word2event)

        used_time = time.time() - start_time
        print(f"token_count={len(generated_words)}, bar_count={bar_count}, used_time={used_time:.2f}s")

    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        logits -= logits.max()
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][0] + 1
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3]  # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word