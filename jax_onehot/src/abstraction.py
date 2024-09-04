from humn import abstraction_model, trainer
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import os
import json
import random

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

        self.data = []

        self.step = 0
        self.epoch_batch = []

        self.loss_alpha = loss_alpha
        self.avg_loss = 0.0


    def accumulate_batch(self, data):
        self.data.append(data)
        return self


    def clear_batch(self):
        self.data = []
        return self
    

    def prepare_batch(self, mini_batch_size):
        self.epoch_batch = []
        for i in range(0, len(self.data), mini_batch_size):
            d = jnp.concatenate(self.data[i:i+mini_batch_size], axis=0)
            self.epoch_batch.append(d)

        # shuffle
        random.shuffle(self.epoch_batch)
        

    def step_update(self):
        minibatch = self.epoch_batch[self.step % len(self.epoch_batch)]
        self.step += 1
        loss = self.model.accumulate(minibatch)
        self.avg_loss = (self.avg_loss * (1-self.loss_alpha) + loss * self.loss_alpha)
        return self.avg_loss


@jax.jit
def compute_maxima_from_stat(stat, first=True, last=True):
    # compute local maxima, include last
    # suppression: maxima is the first down hill after the last maximum
    # False True True True False True False, will be suppressed to False True False False False True False

    down_hill = stat[:-1] > stat[1:]
    peeks = ~down_hill[:-1] & down_hill[1:]
    maxima = jnp.concatenate([jnp.array([first]), peeks, jnp.array([last])])
    return maxima


class Model(abstraction_model.Model):
    
    def __init__(self, model: core.base.Stat_Model):
        self.model = model

        self.trainer = Trainer(model)


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = core.load(metadata["model"])
        return Model(model)
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "model": core.save(self.model)
            }, f)


    def incrementally_learn(self, path: State_Sequence) -> trainer.Trainer:
        return self.trainer.accumulate_batch(path.data)


    def abstract_path(self, path: State_Sequence) -> Tuple[Pointer_Sequence, State_Sequence]:
        stat = self.model.infer(path.data)
        maxima = compute_maxima_from_stat(stat, True, True)
        maxima_indices = jnp.arange(len(maxima))[maxima]
        return Pointer_Sequence(maxima_indices), State_Sequence(path.data[maxima_indices])


    def abstract(self, from_sequence: Augmented_State_Squence, action: Action) -> Tuple[State, Action]:
        stat = self.model.infer(from_sequence.data[:, 0, :])
        maxima = compute_maxima_from_stat(stat, True, False)

        # get the last maximum
        # get last true
        last_maxima_indice = jnp.arange(len(maxima))[maxima][-1]
        return State(from_sequence.data[last_maxima_indice, 0, :]), action


    def specify(self, nl_start: State, nl_action: Union[Action, None] = None, start: Union[State, None] = None) -> Union[Action, State]:
        if nl_action is None:
            return nl_start
        return nl_action



if __name__ == "__main__":

    test = jnp.array([1, 2, 3, 3, 2, 1, 1, 2, 3, 2, 1, 0, -1, 1, 2], dtype=jnp.float32)
    maxima = compute_maxima_from_stat(test)
    print(maxima)
    maxima_indices = jnp.arange(len(maxima))[maxima]
    print(maxima_indices)
    maxima = compute_maxima_from_stat(test, True, False)
    print(maxima)
    last_maxima_indice = jnp.arange(len(maxima))[maxima][-1]
    print(last_maxima_indice)

