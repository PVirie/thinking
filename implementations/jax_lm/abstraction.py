from humn import abstraction_model, trainer
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import os
import json
import random
from functools import partial
from ott.tools.k_means import k_means

try:
    from .algebraic import *
except:
    from algebraic import *


def process_chunk(embedding_chunks, step_size=4):
    length = embedding_chunks.shape[0]
    # split chunks into sub_chunk of step_size
    pivot_size = math.ceil(length/step_size)
    data = jnp.pad(embedding_chunks, ((0, pivot_size * step_size - length), (0, 0)), mode='constant', constant_values=0)
    data = jnp.reshape(data, [pivot_size, step_size, -1])
    # average
    sizes = jnp.ones([pivot_size, 1], jnp.float32) * step_size
    sizes = sizes.at[-1].set(length - (pivot_size - 1) * step_size)
    abstract_chunks = jnp.sum(data, axis=1, keepdims=False) / sizes
    abstract_indices_chunks = [i for i in range(step_size, length, step_size)]
    return abstract_chunks, abstract_indices_chunks
    

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
        output = k_means(data, k=self.k, rng=self.r_key)
        if output.converged:
            self.embeddings = output.centroids


class Model(abstraction_model.Model):

    def __init__(self, input_dims, token_size, skip_steps, r_seed=42):
        self.input_dims = input_dims
        self.token_size = token_size
        self.skip_steps = skip_steps
        self.trainer = KMean_Trainer(self.token_size, r_seed=r_seed)

    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = Model(input_dims=metadata["input_dims"], token_size=metadata["token_size"], skip_steps=metadata["skip_steps"], r_seed=metadata["r_seed"])
        model.trainer.embeddings = jnp.load(os.path.join(path, "embeddings.npy"))
        return model                                               
                                                              

    @staticmethod
    def save(self, path):
        if self.train.embeddings is None:
            raise ValueError("Model has not been trained yet")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "input_dims": self.input_dims,
                "token_size": self.token_size,
                "skip_steps": self.skip_steps,
                "r_seed": self.trainer.r_seed
            }, f)
        jnp.save(os.path.join(path, "embeddings.npy"), self.trainer.embeddings)


    def incrementally_learn(self, path: Embedding_Sequence) -> trainer.Trainer:
        return self.trainer.accumulate_batch(jnp.reshape(path.data, [-1, self.input_dims]))


    def abstract_path(self, path: Embedding_Sequence) -> Tuple[Pointer_Sequence, Embedding_Sequence]:
        abstract_chunks, indices = process_chunk(path.data, step_size=self.skip_steps)
        # now project pivots to embeddings
        # embedding has shape [token_size, input_dims], abstract chunks has shape [pivot_size, input_dims]
        # find max dot product
        max_indices = jnp.argmax(jnp.matmul(abstract_chunks, self.trainer.embeddings.T), axis=1, keepdims=True)

        pivots = jnp.take_along_axis(self.trainer.embeddings, max_indices, axis=0)
        return Pointer_Sequence(indices), Embedding_Sequence(pivots)


    def abstract(self, from_sequence: Augmented_Embedding_Squence, action: Text_Embedding) -> Text_Embedding:
        return Text_Embedding(from_sequence.data[-1, 0, :]), action


    def specify(self, nl_start: Text_Embedding, nl_action: Union[Text_Embedding, None] = None, start: Union[Text_Embedding, None] = None) -> Text_Embedding:
        # find max index
        max_index = jnp.argmax(jnp.matmul(jnp.reshape(nl_action.data, [1, -1]), self.trainer.embeddings.T), axis=1, keepdims=True)
        pivot = jnp.take_along_axis(self.trainer.embeddings, max_index, axis=0)
        
        return Text_Embedding(jnp.reshape(pivot, [self.input_dims]))



if __name__ == "__main__":

    pivots, indices = process_chunk(jnp.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], jnp.float32), 3)

    print(indices)
    print(pivots)

    model = Model(4, 3, 3)
    t = model.incrementally_learn(Embedding_Sequence(jnp.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], jnp.float32)))
    t.train()
    print(t.embeddings)

    indices, pivots = model.abstract_path(Embedding_Sequence(jnp.array([[0.9, 0, 0, 0], [0.9, 0, 0, 0], [0.9, 0, 0, 0], [0, 1.1, 0, 0], [0, 1.1, 0, 0], [0, 1.1, 0, 0], [0, 0, 1.5, 0], [0, 0, 0.7, 0]], jnp.float32)))
    print(indices.data)
    print(pivots.data)

