from humn import abstraction_model
from typing import Tuple

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
        # compute local maxima, include last
        maxima = jnp.concatenate([stat[:-1] > stat[1:], jnp.array([True])])
        maxima_indices = jnp.arange(len(maxima))[maxima]
        return Pointer_Sequence(maxima_indices), State_Sequence(path.data[maxima_indices])


    def abstract(self, from_sequence: State_Sequence, action: Action) -> Tuple[State, Action]:
        stat = self.model.infer(from_sequence.data)
        maxima = jnp.concatenate([stat[:-1] > stat[1:], jnp.array([False])])

        # get the last maximum
        # get last true
        last_maxima_indice = jnp.arange(len(maxima))[maxima][-1]
        return from_sequence[last_maxima_indice], action


    def specify(self, start: State, nl_start: State, nl_action: Action) -> Action:
        return nl_action






if __name__ == "__main__":
    import jax
