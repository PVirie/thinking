from humn import abstraction_model
from typing import Tuple, Union

import jax
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


    def incrementally_learn(self, path: State_Sequence) -> float:
        return self.model.accumulate(path.data)


    def abstract_path(self, path: State_Sequence) -> Tuple[Pointer_Sequence, State_Sequence]:
        stat = self.model.infer(path.data)
        maxima = compute_maxima_from_stat(stat, True, True)
        maxima_indices = jnp.arange(len(maxima))[maxima]
        return Pointer_Sequence(maxima_indices), State_Sequence(path.data[maxima_indices])


    def abstract(self, from_encoding_sequence: Augmented_State_Squence, action: Action) -> Tuple[State, Action]:
        stat = self.model.infer(from_encoding_sequence.data[:, 0, :])
        maxima = compute_maxima_from_stat(stat, True, False)

        # get the last maximum
        # get last true
        last_maxima_indice = jnp.arange(len(maxima))[maxima][-1]
        return from_encoding_sequence[last_maxima_indice, 0, :], action


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

