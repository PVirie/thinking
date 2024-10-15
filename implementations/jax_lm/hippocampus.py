from humn import hippocampus_model, trainer
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np

from ott.tools.k_means import k_means

import os
import json

try:
    from .algebraic import *
except:
    from algebraic import *


class KMean_Trainer(trainer.Trainer):
    def __init__(self, k, r_seed=42):
        self.r_key = jax.random.key(r_seed)
        self.data = []
        self.k = k
        self.embeddings = None

    def accumulate_batch(self, data):
        self.data.append(data)
        return self
    
    def train(self):
        data = jnp.concatenate(self.data, axis=0)
        output = k_means(data, k=min(self.k, data.shape[0]), rng=self.r_key)
        if output.converged:
            self.embeddings = output.centroids

    def manually_prepend(self, embeddings):
        if isinstance(embeddings, List):
            embeddings = jnp.array(embeddings)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = jnp.concatenate([embeddings, self.embeddings], axis=0)


class Model(hippocampus_model.Model):

    def __init__(self, max_length: int, input_dims: int, token_size=256, r_seed=42):
        self.max_length = max_length
        self.input_dims = input_dims
        self.token_size = token_size
        self.data = jnp.zeros((max_length, input_dims), dtype=jnp.float32)
        self.stop_flags = jnp.zeros((max_length, 1), dtype=jnp.float32)
        self.r_seed = r_seed

        self.trainer = KMean_Trainer(self.token_size, r_seed=r_seed)
        self.start = self.max_length
        self.printer = None

        # # more efficient one
        # indices = jnp.arange(0, input_dims, 2)
        # indices = jnp.expand_dims(indices, axis=0)
        # pos = jnp.arange(0, max_length, dtype=jnp.float32)
        # pos = jnp.expand_dims(pos, axis=1)

        # Sin = jnp.sin(pos / 10000 ** (2 * indices / input_dims))
        # Cos = jnp.cos(pos / 10000 ** (2 * indices / input_dims))
        # concat = jnp.stack([Sin, Cos], axis=1)
        # self.positional_encoding = jnp.reshape(concat, (max_length, input_dims))


    def refine(self, chunks):
        if self.trainer.embeddings is None:
            return chunks

        # find max dot product
        max_indices = jnp.argmax(jnp.matmul(chunks, self.trainer.embeddings.T), axis=1, keepdims=True)
        pivots = jnp.take_along_axis(self.trainer.embeddings, max_indices, axis=0)

        if self.printer is not None:
            self.printer(max_indices)

        return pivots


    def encode(self, chunks):
        if self.trainer.embeddings is None:
            raise ValueError("Model has not been trained yet")
        return jnp.argmax(jnp.matmul(chunks, self.trainer.embeddings.T), axis=1)


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = Model(**metadata)
        model.trainer.embeddings = jnp.load(os.path.join(path, "embeddings.npy"))
        return model
                                                              

    @staticmethod
    def save(self, path):
        if self.trainer.embeddings is None:
            raise ValueError("Model has not been trained yet")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "max_length": self.max_length, 
                "input_dims": self.input_dims,
                "token_size": self.token_size,
                "r_seed": self.r_seed
            }, f)
        jnp.save(os.path.join(path, "embeddings.npy"), self.trainer.embeddings)


    def incrementally_learn(self, path: Embedding_Sequence) -> trainer.Trainer:
        return self.trainer.accumulate_batch(jnp.reshape(path.data, [-1, self.input_dims]))


    def augmented_all(self) -> Augmented_Embedding_Squence:
        # data has shape (N, dim + 1)
        return Augmented_Embedding_Squence(
            jnp.concatenate([self.data[self.start:], self.stop_flags[self.start:]], axis=1)
        )


    def augment(self, path: Embedding_Sequence, pivot_sequence: Pointer_Sequence) -> Augmented_Embedding_Squence:
        length = min(self.max_length, path.data.shape[0])
        refined = self.refine(path.data)

        flags = jnp.zeros((path.data.shape[0], 1), dtype=jnp.float32)
        flags = flags.at[pivot_sequence.data].set(100.0)

        return Augmented_Embedding_Squence(
            jnp.concatenate([refined[path.data.shape[0] - length:], flags[path.data.shape[0] - length:]], axis=1)
        )
    

    def append(self, state):
        if isinstance(state, Augmented_Text_Embedding):
            state_data = state.data[:-1]
            flag = state.data[self.input_dims]
        else:
            state_data = state.data
            flag = 0

        # roll the data
        self.data = jnp.roll(self.data, -1, axis=0)
        self.stop_flags = jnp.roll(self.stop_flags, -1, axis=0)
        # inplace update
        refined = self.refine(jnp.reshape(state_data, [1, -1]))
        refined = jnp.reshape(refined, [-1])
        self.data = self.data.at[-1].set(refined)
        self.stop_flags = self.stop_flags.at[-1, 0].set(flag)
        self.start = max(0, self.start - 1)
        return Text_Embedding(refined)


    def refresh(self):
        # self.data = jnp.zeros((self.max_length, self.input_dims), dtype=jnp.float32)
        self.stop_flags = jnp.zeros((self.max_length, 1), dtype=jnp.float32)
        # self.start = self.max_length



if __name__ == "__main__":

    model = Model(16, 4, 4)
    t = model.incrementally_learn(Embedding_Sequence(jnp.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], jnp.float32)))
    t.train()
    print(t.embeddings)

    path = model.augment(
        Embedding_Sequence(jnp.array([[0.9, 0, 0, 0], [0, 1.1, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 0.7]], jnp.float32)),
        Pointer_Sequence(jnp.array([1, 3], jnp.int32))
    )
    print(path.data)

    test_bool = jnp.array([False, False, True, False])
    print("yeah" if (test_bool[:2]).any() else "nope")
    print("yeah" if (test_bool[2:]).any() else "nope")

    