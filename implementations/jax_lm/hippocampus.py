from humn import hippocampus_model, trainer
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np

import os
import json

try:
    from algebraic import *
except:
    from .algebraic import *


class Model(hippocampus_model.Model):

    def __init__(self, max_length: int, input_dims: int):
        self.max_length = max_length
        self.input_dims = input_dims
        self.stop_flags = jnp.zeros((max_length, 1), dtype=jnp.float32)
        self.data = jnp.zeros((max_length, input_dims), dtype=jnp.float32)
        self.start = self.max_length

        # # more efficient one
        # indices = jnp.arange(0, input_dims, 2)
        # indices = jnp.expand_dims(indices, axis=0)
        # pos = jnp.arange(0, max_length, dtype=jnp.float32)
        # pos = jnp.expand_dims(pos, axis=1)

        # Sin = jnp.sin(pos / 10000 ** (2 * indices / input_dims))
        # Cos = jnp.cos(pos / 10000 ** (2 * indices / input_dims))
        # concat = jnp.stack([Sin, Cos], axis=1)
        # self.positional_encoding = jnp.reshape(concat, (max_length, input_dims))


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = Model(**metadata)
        return model
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "max_length": self.max_length,
                "input_dims": self.input_dims
            }, f)


    def augmented_all(self) -> Augmented_Embedding_Squence:
        # data has shape (N, dims + 1)
        return Augmented_Embedding_Squence(
            jnp.concatenate([self.stop_flags[self.start:], self.data[self.start:]], axis=1)
        )


    def augment(self, path: Embedding_Sequence, pivot_sequence: Pointer_Sequence) -> Augmented_Embedding_Squence:
        length = min(self.max_length, path.data.shape[0])

        flags = jnp.zeros((path.data.shape[0], 1), dtype=jnp.float32)
        flags = flags.at[pivot_sequence.data].set(100.0)

        return Augmented_Embedding_Squence(
            jnp.concatenate([flags[-length:], path.data[-length:]], axis=1)
        )
    

    def append(self, state):
        if isinstance(state, Augmented_Text_Embedding):
            flag = state.data[0]
            state_data = state.data[1:]
        else:
            flag = 0
            state_data = state.data

        # roll the data
        self.stop_flags = jnp.roll(self.stop_flags, -1, axis=0)
        self.data = jnp.roll(self.data, -1, axis=0)
        # inplace update
        self.stop_flags = self.stop_flags.at[-1, 0].set(flag)
        self.data = self.data.at[-1, :].set(state_data)
        self.start = max(0, self.start - 1)


    def refresh(self):
        self.stop_flags = self.stop_flags.at[:].set(0.0)
        self.data = self.data.at[:].set(0.0)
        self.start = self.max_length



if __name__ == "__main__":

    model = Model(16, 4)

    path = model.augment(
        Embedding_Sequence(jnp.array([[0.9, 0, 0, 0], [0, 1.1, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 0.7]], jnp.float32)),
        Pointer_Sequence(jnp.array([1, 3], jnp.int32))
    )
    print(path.data)

    test_bool = jnp.array([False, False, True, False])
    print("yeah" if (test_bool[:2]).any() else "nope")
    print("yeah" if (test_bool[2:]).any() else "nope")

    