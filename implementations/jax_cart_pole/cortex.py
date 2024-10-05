from humn import cortex_model, trainer
from typing import Tuple
import random

from functools import partial
import jax.numpy as jnp
import os
import json
import math

try:
    from .algebric import *
except:
    from algebric import *

try:
    import core
except:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    import core


def generate_mask_and_score(pivots, length, diminishing_factor=0.9, pre_steps=1):
    # from states to pivots
    # pivots = jnp.array(pivots, dtype=jnp.int32)
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    pre_pivots = jnp.concatenate([jnp.full([pre_steps], -1, dtype=jnp.int32), pivots[:-pre_steps]], axis=0)
    masks = jnp.logical_and(jnp.expand_dims(pre_pivots, axis=0) < pos, pos <= jnp.expand_dims(pivots, axis=0)).astype(jnp.float32)
    order = jnp.reshape(jnp.arange(0, -length, -1), [-1, 1]) + jnp.expand_dims(pivots, axis=0)

    scores = jnp.power(diminishing_factor, order)
    return jnp.transpose(masks), jnp.transpose(scores)


def deci_ceil(x):
    return int(math.ceil(x * 10) / 10)


class Trainer(trainer.Trainer):
    def __init__(self, model, loss_alpha=0.05):
        self.model = model

        self.s = []
        self.x = []
        self.t = []
        self.scores = []
        self.masks = []

        self.step = 0
        self.epoch_batch = []

        self.loss_alpha = loss_alpha
        self.avg_loss = 0.0


    def accumulate_batch(self, step_discount_factor, use_action, use_reward, path_encoding_sequence: State_Action_Sequence, pivot_indices: Pointer_Sequence, pivots: Expectation_Sequence):

        cart_state_sequence = path_encoding_sequence.data[:, :4]

        s = jnp.tile(jnp.expand_dims(cart_state_sequence, axis=0), (len(pivots), 1, 1))
        t = jnp.tile(jnp.expand_dims(pivots.data, axis=1), (1, len(path_encoding_sequence), 1))

        if use_action:
            cart_action_sequence = path_encoding_sequence.data[:, 4:]
            x = jnp.tile(jnp.expand_dims(cart_action_sequence, axis=0), (len(pivots), 1, 1))
        else:
            x = jnp.tile(jnp.expand_dims(jnp.roll(cart_state_sequence, -1, axis=0), axis=0), (len(pivots), 1, 1))

        # s has shape (P, seq_len, dim), a has shape (P, seq_len, dim), t has shape (P, seq_len, dim), scores has shape (P, seq_len), masks has shape (P, seq_len)
        masks, scores = generate_mask_and_score(pivot_indices.data, len(path_encoding_sequence), step_discount_factor, min(2, pivot_indices.data.shape[0]))
        if use_reward:
            # t has shape (P, seq_len, 1)
            scores = scores * jnp.reshape(t, (len(pivots), len(path_encoding_sequence)))

        # now always pad seq_len to tens
        seq_len = s.shape[1]
        target_seq_len = deci_ceil(seq_len)
        s = jnp.pad(s, ((0, 0), (target_seq_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
        x = jnp.pad(x, ((0, 0), (target_seq_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
        t = jnp.pad(t, ((0, 0), (target_seq_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
        scores = jnp.pad(scores, ((0, 0), (target_seq_len - seq_len, 0)), mode="constant", constant_values=0.0)
        masks = jnp.pad(masks, ((0, 0), (target_seq_len - seq_len, 0)), mode="constant", constant_values=0.0)

        self.s.append(s)
        self.x.append(x)
        self.t.append(t)
        self.scores.append(scores)
        self.masks.append(masks)

        return self


    def clear_batch(self):
        self.s = []
        self.x = []
        self.t = []
        self.scores = []
        self.masks = []

        return self
    

    def prepare_batch(self, mini_batch_size):
        # try grouping batch with the same sequence length
        seq_len_to_batch = {}
        for i in range(len(self.s)):
            seq_len = self.s[i].shape[1]
            if seq_len not in seq_len_to_batch:
                seq_len_to_batch[seq_len] = {
                    "size": 0,
                    "s": [],
                    "x": [],
                    "t": [],
                    "scores": [],
                    "masks": []
                }
            seq_len_to_batch[seq_len]["size"] += 1
            seq_len_to_batch[seq_len]["s"].append(self.s[i])
            seq_len_to_batch[seq_len]["x"].append(self.x[i])
            seq_len_to_batch[seq_len]["t"].append(self.t[i])
            seq_len_to_batch[seq_len]["scores"].append(self.scores[i])
            seq_len_to_batch[seq_len]["masks"].append(self.masks[i])

        self.epoch_batch = []
        for seq_len, group in seq_len_to_batch.items():
            S = jnp.concatenate(group["s"], axis=0)
            X = jnp.concatenate(group["x"], axis=0)
            T = jnp.concatenate(group["t"], axis=0)
            Scores = jnp.concatenate(group["scores"], axis=0)
            Masks = jnp.concatenate(group["masks"], axis=0)

            for i in range(0, group["size"], mini_batch_size):
                s = S[i:i+mini_batch_size]
                x = X[i:i+mini_batch_size]
                t = T[i:i+mini_batch_size]
                scores = Scores[i:i+mini_batch_size]
                masks = Masks[i:i+mini_batch_size]
                self.epoch_batch.append((s, x, t, scores, masks))

        # shuffle
        random.shuffle(self.epoch_batch)
        

    def step_update(self):
        minibatch = self.epoch_batch[self.step % len(self.epoch_batch)]
        self.step += 1
        loss = self.model.fit_sequence(minibatch[0], minibatch[1], minibatch[2], minibatch[3], minibatch[4])
        self.avg_loss = (self.avg_loss * (1-self.loss_alpha) + loss * self.loss_alpha)
        return self.avg_loss


class Model(cortex_model.Model):
    
    def __init__(self, layer: int, return_action: bool, use_reward: bool, model: core.base.Model, step_discount_factor=0.9):
        # if you wish to share the model, pass the index into learning and inference functions to differentiate between layers
        self.layer = layer
        self.return_action = return_action
        self.use_reward = use_reward
        self.step_discount_factor = step_discount_factor
        self.model = model
        self.printer = None

        self.trainer = Trainer(self.model)


    def set_printer(self, printer):
        self.printer = printer


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = core.load(metadata["model"])
        return Model(layer=metadata["layer"], return_action=metadata["return_action"], use_reward=metadata["use_reward"], model=model, step_discount_factor=metadata["step_discount_factor"])
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "layer": self.layer,
                "return_action": self.return_action,
                "use_reward": self.use_reward,
                "step_discount_factor": self.step_discount_factor,
                "model": core.save(self.model)
            }, f)


    def incrementally_learn(self, path_encoding_sequence: State_Action_Sequence, pivot_indices: Pointer_Sequence, pivots: Expectation_Sequence) -> Trainer:
        # learn to predict the next state and its probability from the current state given goal
        return self.trainer.accumulate_batch(self.step_discount_factor, self.return_action, self.use_reward, path_encoding_sequence, pivot_indices, pivots)


    def infer_sub_action(self, from_encoding_sequence: Cart_State, expect_action: Action) -> Action:
        next_action_data, score = self.model.infer(
            jnp.reshape(from_encoding_sequence.data, (1, 1, -1)), 
            jnp.reshape(expect_action.data, (1, -1))
            )
        if self.return_action:
            a = Action(next_action_data[0])
        else:
            a = Expectation(next_action_data[0])
        if self.printer is not None:
            self.printer(from_encoding_sequence + a)
        return a


if __name__ == "__main__":
    import jax