from humn import cortex_model, trainer
from typing import Tuple
import random

from functools import partial
import jax.numpy as jnp
import os
import json

try:
    from algebraic import *
except:
    from .algebraic import *

try:
    import core
except:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    import core


def generate_mask_and_distance(pivots, length, distances, pre_steps=1):
    # from states to pivots
    # pivots = jnp.array(pivots, dtype=jnp.int32)
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    pre_pivots = jnp.concatenate([jnp.full([pre_steps], -1, dtype=jnp.int32), pivots[:-pre_steps]], axis=0)
    masks = jnp.logical_and(jnp.expand_dims(pre_pivots, axis=0) <= pos, pos < jnp.expand_dims(pivots, axis=0)).astype(jnp.float32)
    orders = jnp.reshape(distances[pivots], [1, -1]) - jnp.reshape(distances, [-1, 1])

    return jnp.transpose(masks), jnp.transpose(orders)


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


    def accumulate_batch(self, max_steps, path_encoding_sequence: Augmented_State_Squence, pivot_indices: Pointer_Sequence, distances: Distance_Sequence):

        pivots = path_encoding_sequence[pivot_indices].data[:, 0, :]
        sequence_data = path_encoding_sequence.data[:, 0, :]
        distances = distances.data

        s = jnp.tile(jnp.expand_dims(sequence_data, axis=0), (len(pivots), 1, 1))
        s_ = jnp.tile(jnp.expand_dims(jnp.roll(sequence_data, -1, axis=0), axis=0), (len(pivots), 1, 1))
        x = s_ - s
        t = jnp.tile(jnp.expand_dims(pivots, axis=1), (1, len(path_encoding_sequence), 1))

        # s has shape (P, seq_len, dim), a has shape (P, seq_len, dim), t has shape (P, seq_len, dim), scores has shape (P, seq_len), masks has shape (P, seq_len)
        pivot_look_ahead_steps = min(2, pivot_indices.data.shape[0])
        masks, scores = generate_mask_and_distance(pivot_indices.data, len(path_encoding_sequence), distances, pivot_look_ahead_steps)

        max_gap = max_steps * pivot_look_ahead_steps
        # distance 0 get scores 1, distance max_gap get scores 0, linearly interpolate
        scores = 1 - scores / max_gap

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
    
    def __init__(self, layer: int, model: core.base.Model, max_steps=16):
        # if you wish to share the model, pass the index into learning and inference functions to differentiate between layers
        self.layer = layer
        self.max_steps = max_steps
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
        return Model(layer=metadata["layer"], model=model, max_steps=metadata["max_steps"])
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "layer": self.layer,
                "max_steps": self.max_steps,
                "model": core.save(self.model)
            }, f)


    def incrementally_learn(self, path_encoding_sequence: Augmented_State_Squence, pivot_indices: Pointer_Sequence, pivots: Distance_Sequence) -> Trainer:
        # learn to predict the next state and its probability from the current state given goal
        return self.trainer.accumulate_batch(self.max_steps, path_encoding_sequence, pivot_indices, pivots)


    def infer_sub_action(self, from_encoding_sequence: Augmented_State_Squence, expect_action: Action) -> Action:
        goal_state = from_encoding_sequence + expect_action
        next_action_data, score = self.model.infer(
            jnp.expand_dims(from_encoding_sequence.data[:, 0, :], axis=0), 
            jnp.expand_dims(goal_state.data, axis=0)
            )
        a = Action(next_action_data[0])
        if self.printer is not None:
            self.printer(from_encoding_sequence + a)
        return a


if __name__ == "__main__":
    import jax

    masks, distances = generate_mask_and_distance(jnp.array([0, 3, 7]), 8, jnp.arange(8), 2)
    print(masks)
    print(distances)

    table_model = core.table.Model(4, 1)
    model = Model(0, table_model)
    
    r_key = jax.random.key(42)
    r_key, subkey = jax.random.split(r_key)
    states = Augmented_State_Squence(jax.random.normal(subkey, (10, 2, 4)))

    model.incrementally_learn(states, Pointer_Sequence([5, 9]), Distance_Sequence(jnp.arange(10, dtype=jnp.float32)))