from humn import hippocampus_model
from typing import Tuple
import jax.numpy as jnp

import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from algebric import *



class Model(hippocampus_model.Model):

    def __init__(self, max_length: int, dims: int):
        self.max_length = max_length
        self.input_dims = dims
        self.data = jnp.zeros((max_length, dims), dtype=jnp.float32)

        # make positional encoding
        self.positional_encoding = jnp.zeros((max_length, dims), dtype=jnp.float32)
        # use sin and cos positional encoding
        pos = jnp.arange(0, max_length, dtype=jnp.float32)
        for i in range(0, dims, 2):
            self.positional_encoding[:, i] = jnp.sin(pos / 10000 ** (2 * i / dims))
            self.positional_encoding[:, i + 1] = jnp.cos(pos / 10000 ** (2 * i / dims))



    def __call__(self) -> Augmented_State_Squence:
        # data has shape (N, 2, dim)
        return Augmented_State_Squence(jnp.concatenate([self.data, self.positional_encoding], axis=1))


    def append(self, state: State):
        # roll the data
        self.data = jnp.roll(self.data, -1, axis=0)
        # inplace update
        self.data[-1] = state.data


    def extend(self, path: State_Sequence):
        # roll the data
        self.data = jnp.roll(self.data, -len(path), axis=0)
        # inplace update
        self.data[-len(path):] = path.data


    def refresh(self):
        self.data[:] = 0


