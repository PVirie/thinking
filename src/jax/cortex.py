from humn import cortex_model, trainer
from typing import Tuple

from functools import partial
import jax.numpy as jnp
import os
import json

try:
    from .algebric import *
except:
    from algebric import *

try:
    from .. import core
except:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import core


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


    def accumulate_batch(self, s, x, t, scores, masks):
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
        self.epoch_batch = []
        for i in range(0, len(self.s), mini_batch_size):
            s = jnp.concatenate(self.s[i:i+mini_batch_size], axis=0)
            x = jnp.concatenate(self.x[i:i+mini_batch_size], axis=0)
            t = jnp.concatenate(self.t[i:i+mini_batch_size], axis=0)
            scores = jnp.concatenate(self.scores[i:i+mini_batch_size], axis=0)
            masks = jnp.concatenate(self.masks[i:i+mini_batch_size], axis=0)
            self.epoch_batch.append((s, x, t, scores, masks))


    def step_update(self):
        minibatch = self.epoch_batch[self.step % len(self.epoch_batch)]
        self.step += 1
        loss = self.model.fit(minibatch[0], minibatch[1], minibatch[2], minibatch[3], minibatch[4])
        self.avg_loss = (self.avg_loss * (1-self.loss_alpha) + loss * self.loss_alpha)
        return self.avg_loss


def generate_mask_and_score(pivots, length, diminishing_factor=0.9, pre_steps=1):
    # pivots = jnp.array(pivots, dtype=jnp.int32)
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    
    pre_pivots = jnp.concatenate([jnp.full([pre_steps], -1, dtype=jnp.int32), pivots[:-pre_steps]], axis=0)

    masks = jnp.logical_and(pos > jnp.expand_dims(pre_pivots, axis=0), pos <= jnp.expand_dims(pivots, axis=0)).astype(jnp.float32)

    order = jnp.reshape(jnp.arange(0, -length, -1), [-1, 1]) + jnp.expand_dims(pivots, axis=0)
    scores = jnp.power(diminishing_factor, order)
    return masks, scores


def prepare(step_discount_factor, path_encoding_sequence: Augmented_State_Squence, pivot_indices: Pointer_Sequence, pivots: State_Sequence):
    distances = jnp.arange(len(path_encoding_sequence))

    # learn to predict the next state and its probability from the current state given goal
    masks, scores = generate_mask_and_score(pivot_indices.data, len(path_encoding_sequence), step_discount_factor, min(2, pivot_indices.data.shape[0]))

    pivots = path_encoding_sequence[pivot_indices].data[:, 0, :]
    sequence_data = path_encoding_sequence.data[:, 0, :]

    s = jnp.tile(jnp.expand_dims(sequence_data, axis=1), (1, len(pivots), 1))
    x = jnp.tile(jnp.expand_dims(jnp.roll(sequence_data, -1, axis=0), axis=1), (1, len(pivots), 1))
    a = x - s
    t = jnp.tile(jnp.expand_dims(pivots, axis=0), (len(path_encoding_sequence), 1, 1))

    s = jnp.reshape(s, (-1, s.shape[-1]))
    a = jnp.reshape(a, (-1, a.shape[-1]))
    t = jnp.reshape(t, (-1, t.shape[-1]))
    scores = jnp.reshape(scores, (-1))
    masks = jnp.reshape(masks, (-1))

    # s has shape (N, dim), a has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)
    return s, a, t, scores, masks


class Model(cortex_model.Model):
    
    def __init__(self, layer: int, model: core.base.Model, step_discount_factor=0.9):
        # if you wish to share the model, pass layer into learning and inference functions to differentiate
        self.layer = layer
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
        return Model(layer=metadata["layer"], model=model, step_discount_factor=metadata["step_discount_factor"])
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "layer": self.layer,
                "step_discount_factor": self.step_discount_factor,
                "model": core.save(self.model)
            }, f)


    def incrementally_learn(self, path_encoding_sequence: Augmented_State_Squence, pivot_indices: Pointer_Sequence, pivots: State_Sequence) -> Trainer:
        s, a, t, scores, masks = prepare(self.step_discount_factor, path_encoding_sequence, pivot_indices, pivots)
        # s has shape (N, dim), a has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)
        return self.trainer.accumulate_batch(s, a, t, scores, masks)


    def infer_sub_action(self, from_encoding_sequence: Augmented_State_Squence, expect_action: Action) -> Action:
        goal_state = from_encoding_sequence + expect_action
        next_action_data, score = self.model.infer(
            jnp.expand_dims(from_encoding_sequence.data[-1, 0, :], axis=0), 
            jnp.expand_dims(goal_state.data, axis=0)
            )
        a = Action(next_action_data[0])
        if self.printer is not None:
            self.printer(from_encoding_sequence + a)
        return a


if __name__ == "__main__":
    import jax

    masks, scores = generate_mask_and_score(jnp.array([0, 3, 7]), 8, 0.9, 2)
    print(masks)
    print(scores)

    table_model = core.table.Model(4)
    model = Model(0, table_model)
    
    r_key = jax.random.key(42)
    r_key, subkey = jax.random.split(r_key)
    states = Augmented_State_Squence(jax.random.normal(subkey, (10, 2, 4)))

    model.incrementally_learn(states, Pointer_Sequence([5, 9]), None)