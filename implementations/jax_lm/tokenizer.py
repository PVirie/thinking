from typing import List
import jax
import jax.numpy as jnp
from ott.tools.k_means import k_means
import os
import json

try:
    from algebraic import *
except:
    from .algebraic import *


class KMean_Tokenizer:
    def __init__(self, k, r_seed=42):
        self.r_seed = r_seed
        self.r_key = jax.random.key(r_seed)
        self.data = []
        self.k = k
        self.embeddings = None


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = KMean_Tokenizer(**metadata)
        model.embeddings = jnp.load(os.path.join(path, "embeddings.npy"))
        model.freeze()
        return model
                                                              

    @staticmethod
    def save(self, path):
        if self.embeddings is None:
            raise ValueError("Model has not been trained yet")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "r_seed": self.r_seed, 
                "k": self.k,
            }, f)
        jnp.save(os.path.join(path, "embeddings.npy"), self.embeddings)


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


    def freeze(self):
        self.one_hots = jnp.eye(self.embeddings.shape[0]) * 100


    def output_dims(self):
        return self.embeddings.shape[0]


    def decode(self, codes):
        if len(codes.shape) == 1:
            # codes has shape (D,)
            max_indices = jnp.argmax(codes, axis=0)
            embeddings = self.embeddings[max_indices]
        else:
            # codes has shape (N, D)
            max_indices = jnp.argmax(codes, axis=1, keepdims=True)
            embeddings = jnp.take_along_axis(self.embeddings, max_indices, axis=0)
        return embeddings


    def encode(self, embeddings, return_indices=False):
        if len(embeddings.shape) == 1:
            # embeddings has shape (E,)
            max_indices = jnp.argmax(jnp.matmul(embeddings, self.embeddings.T))
            if return_indices:
                return max_indices
            return self.one_hots[max_indices]
        # embeddings has shape (N, E)
        max_indices = jnp.argmax(jnp.matmul(embeddings, self.embeddings.T), axis=1)
        if return_indices:
            return max_indices
        return self.one_hots[max_indices]
    

if __name__ == "__main__":

    t = KMean_Tokenizer(4)
    t = t.accumulate_batch(jnp.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], jnp.float32))
    t.train()
    t.freeze()
    print(t.embeddings)

    encoded = t.encode(jnp.array([[0.9, 0, 0, 0], [0, 1.1, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 0.7]], jnp.float32))
    print(encoded)

    decoded = t.decode(encoded)
    print(decoded)

    encoded = t.encode(jnp.array([0.9, 0, 0, 0], jnp.float32))
    print(encoded)

    decoded = t.decode(encoded)
    print(decoded)
