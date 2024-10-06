from humn import hippocampus_model
from typing import Tuple
import jax.numpy as jnp

import os
import json

from .algebraic import *


class Model(hippocampus_model.Model):

    def __init__(self, max_length: int, dims: int):
        self.max_length = max_length
        self.input_dims = dims
        self.data = jnp.zeros((max_length, dims), dtype=jnp.float32)

        self.start = self.max_length

        # make positional encoding
        
        # self.positional_encoding = jnp.zeros((max_length, dims), dtype=jnp.float32)
        # use sin and cos positional encoding
        # pos = jnp.arange(0, max_length, dtype=jnp.float32)
        # for i in range(0, dims, 2):
        #     self.positional_encoding = self.positional_encoding.at[:, i].set(jnp.sin(pos / 10000 ** (2 * i / dims)))
        #     self.positional_encoding = self.positional_encoding.at[:, i + 1].set(jnp.cos(pos / 10000 ** (2 * i / dims)))
        
        # more efficient one
        indices = jnp.arange(0, dims, 2)
        indices = jnp.expand_dims(indices, axis=0)
        pos = jnp.arange(0, max_length, dtype=jnp.float32)
        pos = jnp.expand_dims(pos, axis=1)

        Sin = jnp.sin(pos / 10000 ** (2 * indices / dims))
        Cos = jnp.cos(pos / 10000 ** (2 * indices / dims))
        concat = jnp.stack([Sin, Cos], axis=1)
        self.positional_encoding = jnp.reshape(concat, (max_length, dims))


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = Model(metadata["max_length"], metadata["dims"])
        return model
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({"max_length": self.max_length, "dims": self.input_dims}, f)



    def augmented_all(self) -> Augmented_Embedding_Squence:
        # data has shape (N, 2, dim)
        return Augmented_Embedding_Squence(
            jnp.stack([self.data[self.start:], self.positional_encoding[self.start:]], axis=1)
        )


    def augment(self, path: Embedding_Sequence) -> Augmented_Embedding_Squence:
        length = min(self.max_length, path.data.shape[0])
        return Augmented_Embedding_Squence(
            jnp.stack([path.data[path.data.shape[0] - length:], self.positional_encoding[self.max_length - length:]], axis=1)
        )


    def append(self, state: Text_Embedding):
        # roll the data
        self.data = jnp.roll(self.data, -1, axis=0)
        # inplace update
        self.data = self.data.at[-1].set(state.data)
        
        self.start = max(0, self.start - 1)


    def refresh(self):
        self.data = jnp.zeros((self.max_length, self.input_dims), dtype=jnp.float32)
        self.start = self.max_length

