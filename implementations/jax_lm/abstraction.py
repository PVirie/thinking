from humn import abstraction_model, trainer
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import os
import json
import math

try:
    from algebraic import *
except:
    from .algebraic import *


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
    abstract_indices_chunks = [min((i + 1) * step_size, length) for i in range(pivot_size)]
    return abstract_chunks, abstract_indices_chunks
    

class Model(abstraction_model.Model):

    def __init__(self):
        pass

    @staticmethod
    def load(path):
        model = Model()
        return model                                               
                                                              

    @staticmethod
    def save(self, path):
        pass


    def specify(self, nl_action: Augmented_Text_Embedding) -> Text_Embedding:
        return Text_Embedding(nl_action.data[:-1])


if __name__ == "__main__":

    pivots, indices = process_chunk(jnp.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], jnp.float32), 3)

    print(indices)
    print(pivots)

